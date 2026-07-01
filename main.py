"""
drone_human.py  ―  Tello ドローン 人間追従プログラム
======================================================
  ウィンドウ 1: "Tello Camera"   カメラ映像 + 物体検知描画
  ウィンドウ 2: "Tello Dashboard" テキスト情報パネル
  外部 API / ダッシュボードモジュール不使用
"""

from __future__ import annotations

import time
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from threading import Thread, Lock, Event
from typing import Optional

from djitellopy import Tello
from ultralytics import YOLO


# =========================================================
# 設定
# =========================================================
@dataclass
class Config:
    run_flight: bool = True

    # 追従距離
    near_box_ratio: float = 0.85
    far_box_ratio:  float = 0.55

    # 前後速度
    forward_speed:  int = 1000
    backward_speed: int = 1000

    # Yaw PD
    yaw_kp: float = 0.20
    yaw_kd: float = 0.20
    yaw_max: int  = 80

    # 上下 PD
    ud_kp: float = 0.20
    ud_kd: float = 0.20
    ud_max: int  = 45

    # 高度制御
    height_min:     int   = 120
    height_max:     int   = 150
    height_target:  int   = 1
    height_kp:      float = 10
    height_max_spd: int   = 40

    # 検出
    confidence:    float = 0.40
    img_size:      int   = 416
    detect_every:  int   = 2
    pose_conf_thr: float = 0.30

    # 障害物
    obstacle_margin_px: int = 30
    obstacle_close_cm:  int = 10

    # 安全
    low_battery_land: int = 20
    max_search_wait:  int = 30

    # RC 不感帯
    rc_deadband: int = 8

    # ダッシュボードウィンドウサイズ
    dash_w: int = 420
    dash_h: int = 620  # ログ行追加分を少し拡張

    # ---- 見失い再探索 ----
    # 最後の位置から回転する場合の Yaw 速度
    reacquire_yaw_speed:    int   = 100
    # 最後の位置から後退する場合の FB 速度（負値）
    reacquire_back_speed:   int   = -100
    # 画面下寄りと判定する bh_ratio 閾値（これ以上なら「近すぎ」→後退）
    reacquire_bottom_ratio: float = 0.80
    # 「画面端」と判定する X 方向の割合（中心からこの比率以上外れていたら左右回転）
    reacquire_side_ratio:   float = 0.10


CFG = Config()


# =========================================================
# YOLO 初期化（pose モデル優先、なければ検出モデル）
# =========================================================
def _load_person_model() -> tuple[YOLO, bool]:
    try:
        m = YOLO("yolov8n-pose.pt")
        print("[INFO] Person model: yolov8n-pose")
        return m, True
    except Exception:
        m = YOLO("yolov10n.pt")
        print("[WARN] Pose model not found → yolov10n")
        return m, False

person_model, USE_POSE = _load_person_model()


# =========================================================
# ユーティリティ
# =========================================================
def clamp(v, lo, hi):  return max(lo, min(hi, v))
def deadband(v, band): return 0 if abs(v) <= band else v


# =========================================================
# 向き推定（ポーズキーポイントから）
# =========================================================
def estimate_facing(kp_xy, kp_conf) -> str:
    if len(kp_xy) < 7:
        return "unknown"
    def c(i): return float(kp_conf[i]) if i < len(kp_conf) else 0.0
    thr = CFG.pose_conf_thr
    face_score = c(0) + c(1) + c(2)
    ear_score  = c(3) + c(4)
    if face_score > thr * 2:   return "front"
    if ear_score > thr * 1.5 and face_score < thr: return "back"
    if c(5) > thr and c(6) > thr and face_score < thr * 0.5:
        if abs(kp_xy[6][0] - kp_xy[5][0]) > 0: return "back"
    return "unknown"


# =========================================================
# データ型
# =========================================================
@dataclass
class Person:
    id:       int
    box:      tuple
    conf:     float
    facing:   str
    center:   tuple
    bh_ratio: float


# =========================================================
# 検出スレッド（人間のみ）
# =========================================================
class DetectionThread:
    def __init__(self):
        self._lock      = Lock()
        self._frame     = None
        self._new_frame = Event()
        self._persons:  list[Person] = []
        self._running   = True
        self._thread    = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._lock: self._frame = frame.copy()
        self._new_frame.set()

    @property
    def persons(self):
        with self._lock: return list(self._persons)

    def stop(self):
        self._running = False
        self._new_frame.set()

    def _loop(self):
        while self._running:
            self._new_frame.wait(); self._new_frame.clear()
            if not self._running: break
            with self._lock: frame = self._frame
            if frame is None: continue
            with self._lock:
                self._persons = self._detect(frame)

    def _detect(self, frame):
        h, w = frame.shape[:2]
        results = person_model.predict(
            frame, imgsz=CFG.img_size, conf=CFG.confidence,
            classes=[0], verbose=False,
        )
        persons = []
        if not results or results[0].boxes is None:
            return persons
        boxes  = results[0].boxes
        has_kp = (USE_POSE
                  and hasattr(results[0], "keypoints")
                  and results[0].keypoints is not None)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            facing = "unknown"
            if has_kp and i < len(results[0].keypoints.data):
                kp      = results[0].keypoints.data[i].cpu().numpy()
                kp_xy   = [(float(p[0]), float(p[1])) for p in kp]
                kp_conf = ([float(p[2]) for p in kp]
                           if kp.shape[1] == 3 else [1.0] * len(kp))
                facing  = estimate_facing(kp_xy, kp_conf)
            persons.append(Person(
                id=i + 1, box=(x1, y1, x2, y2), conf=float(box.conf[0]),
                facing=facing, center=((x1 + x2) // 2, (y1 + y2) // 2),
                bh_ratio=(y2 - y1) / h,
            ))
        return persons


# =========================================================
# RC コントローラ（定周期送信）
# =========================================================
class RCController:
    def __init__(self, tello):
        self._tello   = tello
        self._lock    = Lock()
        self._running = True
        self._cmd     = [0, 0, 0, 0]
        self._thread  = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set(self, lr, fb, ud, yaw, paused=False):
        if paused: lr = fb = ud = yaw = 0
        vals = [
            deadband(int(clamp(lr,  -100, 100)), CFG.rc_deadband),
            deadband(int(clamp(fb,  -100, 100)), CFG.rc_deadband),
            deadband(int(clamp(ud,  -100, 100)), CFG.rc_deadband),
            deadband(int(clamp(yaw, -100, 100)), CFG.rc_deadband),
        ]
        with self._lock: self._cmd = vals

    def stop(self):  self.set(0, 0, 0, 0)

    def close(self):
        self._running = False
        self.stop()
        self._thread.join(timeout=1)

    def _loop(self):
        while self._running:
            with self._lock: cmd = list(self._cmd)
            try:   self._tello.send_rc_control(*cmd)
            except Exception: pass
            time.sleep(0.08)


# =========================================================
# 制御ロジック
# =========================================================
class TrackingController:
    def __init__(self):
        self._prev_err_x    = 0
        self._prev_height_e = 0
        self._facing_hist   = deque(maxlen=20)

    def update_facing(self, facing): self._facing_hist.append(facing)

    def stable_facing(self, current):
        if len(self._facing_hist) > 3:
            fc = self._facing_hist.count("front")
            bc = self._facing_hist.count("back")
            if fc > bc: return "front"
            if bc > fc: return "back"
        return current

    def compute(self, target, frame_w, frame_h, tof_cm, height_cm):
        # Yaw
        err_x  = target.center[0] - frame_w // 2
        d_err  = err_x - self._prev_err_x
        yaw    = int(clamp(err_x * CFG.yaw_kp + d_err * CFG.yaw_kd,
                           -CFG.yaw_max, CFG.yaw_max))
        self._prev_err_x = err_x

        # 上下
        ud = self._height_ud(height_cm)

        # 前後
        bh = target.bh_ratio
        if   bh > CFG.near_box_ratio: fb = -CFG.backward_speed
        elif bh < CFG.far_box_ratio:  fb =  CFG.forward_speed
        else:                          fb = 0
        if tof_cm is not None and tof_cm < CFG.obstacle_close_cm and fb > 0:
            fb = 0

        return 0, fb, ud, yaw   # lr=0（横移動なし）

    def _height_ud(self, height_cm):
        if height_cm is None:
            self._prev_height_e = 0
            return 0
        err   = CFG.height_target - height_cm
        d_err = err - self._prev_height_e
        self._prev_height_e = err
        spd   = err * CFG.height_kp + d_err * CFG.height_kp * 0.3
        if height_cm < CFG.height_min:
            return int(clamp(spd, 20, CFG.height_max_spd))
        if height_cm > CFG.height_max:
            return int(clamp(spd, -CFG.height_max_spd, -20))
        return int(clamp(spd * 0.5, -CFG.height_max_spd, CFG.height_max_spd))


# =========================================================
# 見失い再探索ステートマシン
# =========================================================
class ReacquireState:
    """
    人間を見失った瞬間の位置をもとに、
    「左右」→ Yaw 回転、「下寄り（近すぎ）」→ 後退 で再探索する。
    再発見したら即座に IDLE に戻る。

    状態遷移：
        IDLE  ─(見失い)→  REACQUIRING
        REACQUIRING ─(再発見)→  IDLE
        REACQUIRING ─(max_search_wait 超過)→  SEARCHING（既存の Yaw スキャン）
    """

    IDLE         = "IDLE"
    REACQUIRING  = "REACQUIRING"

    def __init__(self):
        self.state = self.IDLE
        # 最後に見ていたフレーム内の正規化位置（0.0〜1.0）
        self._last_norm_x: float = 0.5   # 画面中央
        self._last_norm_y: float = 0.5
        self._last_bh_ratio: float = 0.0  # バウンディングボックス高さ比率

    # ----------------------------------------------------------
    # 外部から呼ぶ API
    # ----------------------------------------------------------
    def on_tracking(self, target: "Person", frame_w: int, frame_h: int):
        """追従中は毎フレーム呼んで最終位置を更新する"""
        self._last_norm_x   = target.center[0] / frame_w
        self._last_norm_y   = target.center[1] / frame_h
        self._last_bh_ratio = target.bh_ratio
        if self.state == self.REACQUIRING:
            # 再発見 → IDLE へ即遷移
            self.state = self.IDLE

    def on_lost(self):
        """見失った最初のフレームで呼ぶ（既に REACQUIRING なら無視）"""
        if self.state == self.IDLE:
            self.state = self.REACQUIRING

    def is_reacquiring(self) -> bool:
        return self.state == self.REACQUIRING

    def get_rc(self) -> tuple[int, int, int, int]:
        """
        REACQUIRING 中に送るべき RC 値を返す。
        優先順位：
          1. 下寄り（bh_ratio が大きい＝近すぎ）→ 後退
          2. 右寄り → 右 Yaw
          3. 左寄り → 左 Yaw
          4. 中央付近 → 緩い右 Yaw（デフォルト探索）
        """
        if self.state != self.REACQUIRING:
            return 0, 0, 0, 0

        lr = fb = ud = yaw = 0

        # ---- 後退優先（最後に近距離検出していた場合）----
        if self._last_bh_ratio >= CFG.reacquire_bottom_ratio:
            fb = CFG.reacquire_back_speed   # 負値 = 後退
            return lr, fb, ud, yaw

        # ---- 左右回転 ----
        deviation = self._last_norm_x - 0.5   # -0.5(左端) 〜 +0.5(右端)
        if abs(deviation) >= CFG.reacquire_side_ratio:
            # 最後にいた方向へ Yaw 回転
            yaw = CFG.reacquire_yaw_speed if deviation > 0 else -CFG.reacquire_yaw_speed
        else:
            # 中央寄りで見失った場合は緩い右旋回
            yaw = CFG.reacquire_yaw_speed

        return lr, fb, ud, yaw

    def get_description(self) -> str:
        """ダッシュボード表示用テキスト"""
        if self.state == self.IDLE:
            return "---"
        deviation = self._last_norm_x - 0.5
        if self._last_bh_ratio >= CFG.reacquire_bottom_ratio:
            return f"後退中 (bh={self._last_bh_ratio:.2f})"
        side = "右" if deviation > 0 else "左"
        return f"{side}回転中 (x={self._last_norm_x:.2f})"


# =========================================================
# 描画ヘルパー  ―  カメラウィンドウ
# =========================================================
FACING_COLORS = {
    "front":   (0, 255, 150),
    "back":    (0, 150, 255),
    "unknown": (180, 180, 180),
}

def draw_persons(frame, persons, target_id):
    for p in persons:
        x1, y1, x2, y2 = p.box
        col   = FACING_COLORS.get(p.facing, (200, 200, 200))
        thick = 3 if p.id == target_id else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
        label = f"#{p.id} {p.facing} {p.conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)
        if p.id == target_id:
            cv2.putText(frame, "TRACKING", (x1 + 3, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

def draw_camera_hud(frame, mode, tof, height, paused, selected_id,
                    n_persons, fps, fb, yaw, ud, reacquire_desc="---"):
    h, w = frame.shape[:2]
    # 十字線
    cv2.line(frame, (w // 2, 0),   (w // 2, h),   (255, 255, 255), 1)
    cv2.line(frame, (0, h // 2),   (w, h // 2),   (255, 255, 255), 1)

    # モード
    mc = {"Tracking":     (0, 255, 200),
          "Standby":      (0, 200, 255),
          "Reacquiring":  (0, 200, 255),
          "STOPPED":      (0, 0, 255)}.get(mode, (200, 200, 200))
    cv2.putText(frame, f"[{mode}]", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, mc, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS:{fps:.1f}  persons:{n_persons}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)

    # 再探索中は動作説明を表示
    if mode == "Reacquiring":
        cv2.putText(frame, f"Reacq: {reacquire_desc}",
                    (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1)

    # RC 値
    cv2.putText(frame, f"fb={fb:+4d}  yaw={yaw:+4d}  ud={ud:+4d}",
                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 0), 1)

    # ToF / 高度
    if tof is not None:
        tc = (0, 0, 255) if tof < 30 else (255, 255, 0)
        cv2.putText(frame, f"ToF:{tof}cm", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, tc, 1)
    if height is not None:
        hc = (0, 0, 255) if (height < CFG.height_min or height > CFG.height_max) \
             else (255, 255, 0)
        cv2.putText(frame, f"H:{height}cm", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, hc, 1)

    # 強制停止オーバーレイ
    if paused:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 200), 6)
        cv2.putText(frame, "FORCE STOP  [SPACE to resume]",
                    (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3, cv2.LINE_AA)

    # 操作ガイド（最下行）
    sel = "auto" if selected_id == 0 else f"#{selected_id}"
    guide = f"ESC/Q:Land  SPACE:Stop  0-9:Target({sel})"
    cv2.putText(frame, guide, (10, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)


# =========================================================
# ダッシュボードウィンドウ描画
# =========================================================
def make_dashboard(state: dict) -> np.ndarray:
    W, H = CFG.dash_w, CFG.dash_h
    img  = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (18, 18, 24)   # 暗い背景

    def txt(text, x, y, scale=0.48, color=(210, 210, 210), bold=1):
        cv2.putText(img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)

    def bar(label, value, vmin, vmax, x, y, w=200, h=14,
            lo_bad=None, hi_bad=None):
        txt(label, x, y - 2, 0.40, (160, 160, 160))
        ratio = clamp((value - vmin) / max(vmax - vmin, 1), 0, 1)
        bw    = int(w * ratio)
        cv2.rectangle(img, (x, y + 2), (x + w, y + h + 2), (40, 40, 50), -1)
        if   (lo_bad is not None and value <= lo_bad): col = (0, 60, 220)
        elif (hi_bad is not None and value >= hi_bad): col = (0, 60, 220)
        else:                                          col = (0, 180, 100)
        if bw > 0:
            cv2.rectangle(img, (x, y + 2), (x + bw, y + h + 2), col, -1)
        txt(f"{value}", x + w + 6, y + h, 0.40, (220, 220, 220))

    # ---- タイトル ----
    cv2.rectangle(img, (0, 0), (W, 36), (28, 28, 38), -1)
    txt("TELLO  DRONE  DASHBOARD", 12, 24, 0.58, (80, 220, 160), 2)

    # ---- フライト時間 ----
    elapsed  = time.time() - state.get("flight_start", time.time())
    m, s     = divmod(int(elapsed), 60)
    mode_col = {"Tracking":    (0, 200, 130),
                "Standby":     (0, 160, 220),
                "Reacquiring": (0, 160, 220),
                "STOPPED":     (0, 60, 220)}.get(state.get("mode", ""), (180, 180, 180))
    txt(f"{state.get('mode','---')}", 12, 62, 0.60, mode_col, 2)
    txt(f"Flight  {m:02d}:{s:02d}", 160, 62, 0.50, (160, 160, 160))

    y = 80
    cv2.line(img, (8, y), (W - 8, y), (50, 50, 60), 1)

    # ---- バッテリー ----
    y += 22
    bat = state.get("battery", 0)
    bat_col = (0, 200, 80) if bat > 40 else (0, 120, 220) if bat > 20 else (0, 40, 200)
    txt(f"Battery", 12, y, 0.42, (160, 160, 160))
    txt(f"{bat}%", 100, y, 0.55, bat_col, 2)
    bar("", bat, 0, 100, 160, y - 12, 200, 14,
        lo_bad=CFG.low_battery_land)

    # ---- 高度 / ToF ----
    y += 30
    alt = state.get("altitude_cm", 0)
    tof = state.get("tof_cm")
    alt_col = (0, 200, 80) if CFG.height_min <= alt <= CFG.height_max else (0, 80, 220)
    txt(f"Altitude  {alt} cm", 12, y, 0.46, alt_col)
    bar("", alt, 0, 250, 160, y - 12, 200, 14,
        lo_bad=CFG.height_min - 1, hi_bad=CFG.height_max + 1)
    y += 26
    tof_str = f"{tof} cm" if tof is not None else "---"
    tof_col = (0, 80, 220) if (tof is not None and tof < 30) else (200, 200, 200)
    txt(f"ToF (front)  {tof_str}", 12, y, 0.46, tof_col)

    # ---- RC 値 ----
    y += 30
    cv2.line(img, (8, y), (W - 8, y), (50, 50, 60), 1)
    y += 18
    txt("RC  CONTROL", 12, y, 0.44, (120, 180, 240), 1)

    def rc_bar(label, val, bx, by):
        txt(f"{label}", bx, by, 0.38, (140, 140, 160))
        ratio  = clamp((val + 100) / 200, 0, 1)
        bw_max = 80
        bw     = int(bw_max * ratio)
        cv2.rectangle(img, (bx, by + 3), (bx + bw_max, by + 13), (38, 38, 48), -1)
        cv2.rectangle(img, (bx + bw_max // 2, by + 4),
                      (bx + bw_max // 2, by + 12), (80, 80, 90), 1)
        col = (0, 200, 120) if val >= 0 else (0, 120, 200)
        if bw > 0:
            a = bx + bw_max // 2
            b_ = bx + bw
            cv2.rectangle(img, (min(a, b_), by + 3),
                          (max(a, b_), by + 13), col, -1)
        txt(f"{val:+4d}", bx + bw_max + 4, by + 12, 0.36, (200, 200, 200))

    y += 18
    rc_bar("LR  ", state.get("rc_lr",  0), 12,  y)
    rc_bar("FB  ", state.get("rc_fb",  0), 160, y)
    y += 26
    rc_bar("UD  ", state.get("rc_ud",  0), 12,  y)
    rc_bar("YAW ", state.get("rc_yaw", 0), 160, y)

    # ---- ターゲット情報 ----
    y += 32
    cv2.line(img, (8, y), (W - 8, y), (50, 50, 60), 1)
    y += 18
    txt("TARGET", 12, y, 0.44, (120, 180, 240))
    n_p = state.get("n_persons", 0)
    txt(f"persons detected: {n_p}", 120, y, 0.42,
        (0, 200, 130) if n_p > 0 else (120, 120, 120))

    y += 22
    tid = state.get("target_id", 0)
    if tid:
        txt(f"ID: #{tid}  facing: {state.get('target_facing','---')}  "
            f"({state.get('target_facing_stable','---')})",
            12, y, 0.42, (200, 230, 200))
        y += 20
        ex = state.get("target_err_x", 0)
        ey = state.get("target_err_y", 0)
        txt(f"err_x={ex:+4d}  err_y={ey:+4d}  ratio={state.get('target_ratio',0):.2f}",
            12, y, 0.40, (180, 180, 180))
    else:
        txt("No target", 12, y, 0.44, (100, 100, 120))
        y += 20
        # 再探索中の動作説明
        reacq = state.get("reacquire_desc", "---")
        if reacq != "---":
            txt(f"Reacq: {reacq}", 12, y, 0.40, (0, 200, 255))

    # ---- FPS / フレーム数 ----
    y += 32
    cv2.line(img, (8, y), (W - 8, y), (50, 50, 60), 1)
    y += 18
    txt(f"FPS: {state.get('fps', 0.0):.1f}    Frame: {state.get('frame_count', 0)}",
        12, y, 0.42, (140, 140, 160))

    # ---- ログ ----
    y += 26
    cv2.line(img, (8, y), (W - 8, y), (50, 50, 60), 1)
    y += 18
    txt("LOG", 12, y, 0.44, (120, 180, 240))
    y += 18
    log_entries = state.get("log_entries", [])
    for i, entry in enumerate(log_entries[:8]):
        col = (160, 200, 160) if i == 0 else (110, 110, 130)
        txt(entry, 12, y, 0.36, col)
        y += 17
        if y > H - 10: break

    return img


# =========================================================
# センサー
# =========================================================
def get_tof(tello):
    try:
        d = tello.get_distance_tof()
        return int(d) if d and 0 < int(d) <= 500 else None
    except Exception: return None

def get_height(tello):
    try:
        v = tello.get_height()
        return int(v) if v and 0 < int(v) <= 300 else None
    except Exception: return None

def choose_target(persons, frame_w, selected_id):
    if not persons: return None
    if selected_id > 0:
        for p in persons:
            if p.id == selected_id: return p
    cx = frame_w // 2
    return min(persons, key=lambda p: abs(p.center[0] - cx))


# =========================================================
# メインループ
# =========================================================
def main():
    tello = Tello()
    tello.connect()
    bat = tello.get_battery()
    print(f"[INFO] Battery: {bat}%")
    tello.streamon()
    frame_read = tello.get_frame_read()

    is_flying    = False
    force_stop   = False
    selected_id  = 0
    search_start = None
    flight_start = time.time()
    log_entries: list[str] = []

    detector   = DetectionThread()
    controller = TrackingController()
    reacquire  = ReacquireState()          # ← 新規追加
    rc         = RCController(tello)

    # 前フレームのターゲット有無フラグ（見失い検出用）
    prev_had_target = False

    frame_cnt = 0
    fps_t     = time.time()
    fps_cnt   = 0
    fps_val   = 0.0

    # ウィンドウ初期化
    cv2.namedWindow("Tello Camera",    cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tello Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tello Dashboard", CFG.dash_w, CFG.dash_h)

    # RC 状態（ダッシュボード表示用）
    cur_lr = cur_fb = cur_ud = cur_yaw = 0

    def log(msg: str):
        elapsed = time.time() - flight_start
        m, s    = divmod(int(elapsed), 60)
        entry   = f"{m:02d}:{s:02d}  {msg}"
        log_entries.insert(0, entry)
        if len(log_entries) > 40: log_entries.pop()
        print(f"[LOG] {entry}")

    try:
        if CFG.run_flight:
            print("[INFO] Takeoff...")
            tello.takeoff()
            is_flying    = True
            flight_start = time.time()
            log("離陸完了")
            time.sleep(2.5)
            rc.set(0, 0, 25, 0); time.sleep(1.2); rc.stop(); time.sleep(0.12)
            log(f"ホバリング開始 / battery={bat}%")
        else:
            print("[INFO] DRY RUN mode")
            log("DRY RUN モード開始")

        while True:
            # ---- キー入力 ----
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                log("着陸コマンド入力"); break
            elif key == ord(" "):
                force_stop = not force_stop
                log(f"強制停止 {'ON' if force_stop else 'OFF'}")
                if force_stop: rc.stop()
            elif ord("0") <= key <= ord("9"):
                selected_id = key - ord("0")
                log(f"追従対象 → {'auto' if selected_id==0 else f'#{selected_id}'}")

            # ---- バッテリー（10フレームごとに読む）----
            if frame_cnt % 10 == 0:
                bat = tello.get_battery()
                if bat < CFG.low_battery_land:
                    log(f"低バッテリー({bat}%) 自動着陸"); break

            # ---- フレーム取得 ----
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.02); continue
            h, w = frame.shape[:2]
            frame_cnt += 1

            # ---- FPS ----
            fps_cnt += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps_val = fps_cnt / (now - fps_t)
                fps_cnt = 0; fps_t = now

            # ---- 検出 ----
            if frame_cnt % CFG.detect_every == 0:
                detector.submit(frame)
            persons = detector.persons
            target  = choose_target(persons, w, selected_id)
            tid     = target.id if target else None

            tof_v = get_tof(tello)
            h_v   = get_height(tello)

            display = frame.copy()

            # =========================================================
            # 見失い / 再発見 の検出
            # =========================================================
            just_lost  = prev_had_target and (target is None)
            just_found = (not prev_had_target) and (target is not None)

            if just_lost:
                reacquire.on_lost()
                log(f"人間を見失い → 再探索開始 ({reacquire.get_description()})")

            if target is not None:
                # 追従中は常に最終位置を更新（再発見時も含む）
                reacquire.on_tracking(target, w, h)
                if just_found and not reacquire.is_reacquiring():
                    log("人間を再発見 → 追従再開")

            prev_had_target = target is not None

            # =========================================================
            # 制御分岐
            # =========================================================
            if force_stop:
                # ---- 強制停止 ----
                rc.stop()
                cur_lr = cur_fb = cur_ud = cur_yaw = 0
                draw_persons(display, persons, tid)
                draw_camera_hud(display, "STOPPED", tof_v, h_v, True,
                                selected_id, len(persons), fps_val, 0, 0, 0)
                mode_str = "STOPPED"

            elif target is not None:
                # ---- 通常追従 ----
                search_start = None
                controller.update_facing(target.facing)
                lr, fb, ud, yaw = controller.compute(target, w, h, tof_v, h_v)
                rc.set(lr, fb, ud, yaw)
                cur_lr, cur_fb, cur_ud, cur_yaw = lr, fb, ud, yaw
                cv2.line(display, (w // 2, h // 2), target.center, (0, 255, 255), 1)
                draw_persons(display, persons, tid)
                draw_camera_hud(display, "Tracking", tof_v, h_v, False,
                                selected_id, len(persons), fps_val, fb, yaw, ud)
                mode_str = "Tracking"

            elif reacquire.is_reacquiring():
                # ---- 位置ベース再探索 ----
                lr, fb, ud, yaw = reacquire.get_rc()
                rc.set(lr, fb, ud, yaw)
                cur_lr, cur_fb, cur_ud, cur_yaw = lr, fb, ud, yaw
                reacq_desc = reacquire.get_description()
                draw_camera_hud(display, "Reacquiring", tof_v, h_v, False,
                                selected_id, 0, fps_val, fb, yaw, ud,
                                reacquire_desc=reacq_desc)
                mode_str = "Reacquiring"

                # max_search_wait を超えたらランダムスキャンへ移行
                if search_start is None:
                    search_start = time.time()
                wait = time.time() - search_start
                if wait > CFG.max_search_wait:
                    # ReacquireState を IDLE に戻して既存スキャンへ
                    reacquire.state = ReacquireState.IDLE
                    log("再探索タイムアウト → Yaw スキャンへ移行")
                    search_start = None

            else:
                # ---- 通常待機（max_search_wait 後の広域 Yaw スキャン）----
                rc.stop()
                cur_lr = cur_fb = cur_ud = cur_yaw = 0
                if search_start is None:
                    search_start = time.time(); log("人間未検出 → 待機開始")
                wait = time.time() - search_start
                if wait > CFG.max_search_wait:
                    rc.set(0, 0, 0, 25); cur_yaw = 25
                draw_camera_hud(display, "Standby", tof_v, h_v, False,
                                selected_id, 0, fps_val, 0, 0, 0)
                mode_str = "Standby"

            # ---- カメラウィンドウ表示 ----
            cv2.imshow("Tello Camera", display)

            # ---- ダッシュボードウィンドウ表示 ----
            stable = controller.stable_facing(target.facing) if target else "---"
            dash_state = dict(
                battery=bat, altitude_cm=h_v or 0, tof_cm=tof_v,
                rc_lr=cur_lr, rc_fb=cur_fb, rc_ud=cur_ud, rc_yaw=cur_yaw,
                target_id=tid or 0,
                target_err_x=(target.center[0] - w // 2) if target else 0,
                target_err_y=(target.center[1] - h // 2) if target else 0,
                target_ratio=target.bh_ratio if target else 0.0,
                target_facing=target.facing if target else "---",
                target_facing_stable=stable,
                n_persons=len(persons),
                mode=mode_str,
                fps=fps_val, frame_count=frame_cnt,
                flight_start=flight_start, log_entries=list(log_entries),
                reacquire_desc=reacquire.get_description(),
            )
            cv2.imshow("Tello Dashboard", make_dashboard(dash_state))

    finally:
        print("[INFO] Cleanup...")
        log("シャットダウン")
        rc.stop(); detector.stop(); rc.close()
        if is_flying:
            try: tello.land()
            except Exception as e: print("[ERROR] Land:", e)
        for fn in (tello.streamoff, tello.end):
            try:   fn()
            except Exception: pass
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
