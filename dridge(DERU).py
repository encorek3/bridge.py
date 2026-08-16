
"""

from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum
from typing import List, Optional


# ---------------------------------------------------------------------------
# 1. 台灣 DERU 評定法
#    參考: 交通部「公路橋梁檢測及補強規範」、「橋梁定期檢測評等準則」
#    D=劣化程度(Degree) E=劣化範圍(Extent) R=影響性(Relevancy) U=急迫性(Urgency)
# ---------------------------------------------------------------------------
@dataclass
class DERURating:
    """
    台灣橋梁定期檢測評等準則之 DERU 評等
    D, E, R, U 皆為 0~4 級距,數字越大代表劣化/風險/急迫程度越高
    """
    degree: int       # D: 劣化程度 0~4
    extent: int        # E: 劣化範圍 0~4
    relevancy: int       # R: 對結構安全性/服務性之影響度 0~4
    urgency: int           # U: 維修急迫性 0~4

    def __post_init__(self):
        for name, val in [("D", self.degree), ("E", self.extent),
                           ("R", self.relevancy), ("U", self.urgency)]:
            if not 0 <= val <= 4:
                raise ValueError(f"DERU 的 {name} 值須介於 0~4,實際輸入為 {val}")

    @property
    def repair_deadline_note(self) -> str:
        """依台灣「橋梁定期檢測評等準則」慣例,依 U 值判斷處置時限"""
        if self.urgency >= 4:
            return "立即通報緊急處置 (U=4)"
        if self.urgency == 3:
            return "須於 1 年內完成修復 (U=3)"
        return "納入修繕名單,依排程於 3 年內完成修繕 (U<=2)"

    def as_code(self) -> str:
        return f"D{self.degree}E{self.extent}R{self.relevancy}U{self.urgency}"


# ---------------------------------------------------------------------------
# 2. 橋樑構件分類 與 檢測類別
# ---------------------------------------------------------------------------
class BridgeComponent(IntEnum):
    DECK = 1            # 橋面
    SUPERSTRUCTURE = 2  # 上部結構(主梁、桁架等)
    SUBSTRUCTURE = 3    # 下部結構(橋墩、橋台、基礎)
    CULVERT = 4         # 涵管(如適用)


class TaiwanInspectionType(IntEnum):
    """交通部規範之檢測類別"""
    ROUTINE = 1        # 定期檢測: 每 2 年一次
    DAILY_PATROL = 2   # 日常巡查: 每年 4 次
    SPECIAL = 3         # 特別檢測: 視個案判斷(如地震、颱風、事故後)
    DETAILED = 4         # 詳細檢測: 針對定期檢測發現的疑慮構件深入檢測


class InspectionMethod(IntEnum):
    """
    檢測執行方式
    傳統目視檢測之外,近年業界已導入 AI 結合無人機的智慧檢測技術,
    可補足人力難以到達構件(如高橋墩、跨河箱梁底部)的檢測死角。
    參考案例: 黎明工程顧問股份有限公司「AI無人機智慧橋梁檢測系統」
             (113年度研究成果, https://www.limi.com.tw/project_13_30.aspx)
    """
    VISUAL_MANUAL = 1     # 傳統人工目視檢測
    AI_DRONE = 2            # AI 無人機智慧檢測(影像辨識輔助劣化判釋)
    NDT = 3                   # 非破壞性檢測(超音波、染色滲透等)


# ---------------------------------------------------------------------------
# 3. 缺陷記錄 與 檢測紀錄
# ---------------------------------------------------------------------------
@dataclass
class DefectRecord:
    """單一缺陷(劣化現象)記錄,以 DERU 評等呈現"""
    component: BridgeComponent
    location: str
    defect_type: str            # e.g. "疑似疲勞裂縫", "混凝土剝落", "鋼材腐蝕"
    deru: DERURating
    method: InspectionMethod = InspectionMethod.VISUAL_MANUAL  # 本次發現該缺陷所用的檢測方式
    requires_ndt: bool = False    # 是否需安排非破壞性檢測 (超音波/染色滲透等)
    ndt_method: Optional[str] = None   # e.g. "染色滲透探傷", "超音波檢測"
    image_path: Optional[str] = None   # 若由 AI 無人機發現,保留原始影像路徑供覆核追溯
    ai_confidence: Optional[float] = None  # AI 初判信心值 0~1(僅供參考,非官方評等)
    human_reviewed: bool = True    # 是否已由檢測人員複核確認 DERU(規範要求最終評等須人工確認)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "component": self.component.name,
            "location": self.location,
            "defect_type": self.defect_type,
            "deru": self.deru.as_code(),
            "repair_deadline": self.deru.repair_deadline_note,
            "method": self.method.name,
            "requires_ndt": self.requires_ndt,
            "ndt_method": self.ndt_method,
            "image_path": self.image_path,
            "ai_confidence": self.ai_confidence,
            "human_reviewed": self.human_reviewed,
            "note": self.note,
        }


@dataclass
class BridgeInspectionRecord:
    """一次完整檢測記錄"""
    bridge_id: str
    inspection_type: TaiwanInspectionType
    inspection_date: date
    inspector: str
    defects: List[DefectRecord] = field(default_factory=list)

    def add_defect(self, defect: DefectRecord):
        self.defects.append(defect)

    def max_urgency(self) -> int:
        """取所有缺陷中最高的 U 值,用以判斷是否需加速複查/處置"""
        if not self.defects:
            return 0
        return max(d.deru.urgency for d in self.defects)

    def priority_repair_defects(self) -> List[DefectRecord]:
        """依桃園市等地方作法,U >= 3 者列為優先修繕對象"""
        return [d for d in self.defects if d.deru.urgency >= 3]

    def summary(self) -> str:
        lines = [
            f"橋樑編號: {self.bridge_id}",
            f"檢測類型: {self.inspection_type.name}",
            f"檢測日期: {self.inspection_date}",
            f"檢測人員: {self.inspector}",
            f"缺陷數量: {len(self.defects)}",
            f"最高急迫性 U 值: {self.max_urgency()}",
        ]
        for d in self.defects:
            lines.append(
                f"  - [{d.component.name}] {d.location}: {d.defect_type} "
                f"({d.deru.as_code()}) -> {d.deru.repair_deadline_note} "
                f"檢測方式={d.method.name} NDT需求={d.requires_ndt}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. AI 無人機影像初篩模組
#    流程: 無人機空拍影像 -> AI裂縫/剝落偵測(初篩) -> 產生候選缺陷
#         -> 檢測人員複核給予正式 DERU 評等 -> 寫入 DefectRecord
#    注意: 依規範,最終 DERU 評等須由合格檢測人員判定,AI 僅作初步篩選輔助,
#         不能取代人工複核。
# ---------------------------------------------------------------------------
@dataclass
class DroneDetectionCandidate:
    """AI 對單張無人機影像的初步偵測結果(尚未經人工複核,不能直接當作正式缺陷紀錄)"""
    image_path: str
    component: BridgeComponent
    location_hint: str          # 依飛行航點/GPS或人工標註推得的大略位置
    detected_count: int          # AI 偵測到的可疑裂縫/缺陷數量
    confidence: float              # AI 平均信心值 0~1
    annotated_image_path: str        # 標註後(框選可疑區域)的輸出影像路徑


def analyze_drone_image_for_cracks(image_path: str, component: BridgeComponent,
                                    location_hint: str,
                                    output_dir: str = "drone_ai_results") -> DroneDetectionCandidate:
    """
    對單張無人機拍攝影像執行 AI 裂縫初篩。
    底層沿用 Canny 邊緣偵測 + 形態學處理的裂縫偵測邏輯(見專案初版程式碼),
    實務上建議替換為訓練過的 CNN/分割模型以提升準確度。

    需要安裝: pip install opencv-python numpy
    """
    import os
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"無法讀取無人機影像: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    confidences = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        if aspect_ratio > 3 and area > 30:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            count += 1
            # 簡化信心值估算:長寬比與面積正規化(僅供初篩排序用,非嚴謹機率)
            confidences.append(min(1.0, (aspect_ratio / 10) * min(1.0, area / 500)))

    os.makedirs(output_dir, exist_ok=True)
    annotated_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(annotated_path, img)

    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    return DroneDetectionCandidate(
        image_path=image_path,
        component=component,
        location_hint=location_hint,
        detected_count=count,
        confidence=avg_confidence,
        annotated_image_path=annotated_path,
    )


def batch_analyze_drone_folder(folder_path: str, component: BridgeComponent,
                                output_dir: str = "drone_ai_results",
                                extensions: tuple = (".jpg", ".jpeg", ".png")
                                ) -> List[DroneDetectionCandidate]:
    """
    批次處理一整個資料夾的無人機照片,對每張跑 AI 初篩,
    並依「偵測數量、AI信心值」由高到低排序,方便檢測人員優先複核最可疑的照片。

    使用情境: 無人機一次飛完拍了幾十張甚至上百張照片,
             不需要逐張手動呼叫 analyze_drone_image_for_cracks,
             跑一次這個函式就能拿到整批排序好的候選清單。

    參數:
        folder_path: 存放無人機照片的資料夾路徑
        component:   這批照片對應的構件(若一個資料夾混雜多構件,
                     建議依構件分開資料夾,分別呼叫本函式)
        output_dir:  標註後影像的輸出資料夾
        extensions:  要處理的圖片副檔名

    回傳: 依風險程度(偵測數量*信心值)由高到低排序的候選清單
    """
    import os

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"找不到資料夾: {folder_path}")

    image_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(extensions)
    )
    if not image_files:
        print(f"[警告] {folder_path} 內沒有找到符合副檔名 {extensions} 的照片")
        return []

    candidates: List[DroneDetectionCandidate] = []
    for i, filename in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, filename)
        try:
            candidate = analyze_drone_image_for_cracks(
                image_path=image_path,
                component=component,
                location_hint=f"{component.name} - {filename}",  # 建議另外用航點/GPS對照表補上精確位置
                output_dir=output_dir,
            )
            candidates.append(candidate)
            print(f"[{i}/{len(image_files)}] {filename}: "
                  f"偵測到 {candidate.detected_count} 處可疑區域,信心值 {candidate.confidence}")
        except Exception as e:
            print(f"[{i}/{len(image_files)}] {filename}: 分析失敗 - {e}")

    # 依「偵測數量 x 信心值」由高到低排序,風險越高的照片排越前面
    candidates.sort(key=lambda c: c.detected_count * c.confidence, reverse=True)
    return candidates


def print_review_priority_list(candidates: List[DroneDetectionCandidate], top_n: int = 10):
    """列印優先複核清單,方便檢測人員知道該先看哪幾張照片"""
    print(f"\n=== 優先複核清單(前 {min(top_n, len(candidates))} 筆,依風險程度排序) ===")
    for rank, c in enumerate(candidates[:top_n], start=1):
        print(f"{rank}. {c.image_path} "
              f"-> 可疑區域數:{c.detected_count}, 信心值:{c.confidence}, "
              f"標註圖:{c.annotated_image_path}")


def confirm_defect_from_drone_candidate(candidate: DroneDetectionCandidate,
                                         defect_type: str,
                                         deru: DERURating,
                                         requires_ndt: bool = False,
                                         ndt_method: Optional[str] = None,
                                         reviewer_note: str = "") -> DefectRecord:
    """
    檢測人員複核 AI 候選結果後,正式轉為 DefectRecord。
    DERU 由人工依現場判斷輸入 -- 這一步不可省略,AI 信心值僅供參考排序,
    不可直接當作 D/E/R/U 的評等依據。
    """
    note = (f"AI無人機初篩偵測到 {candidate.detected_count} 處可疑區域"
            f"(平均信心值 {candidate.confidence}),影像:{candidate.annotated_image_path}。"
            f"{reviewer_note}")
    return DefectRecord(
        component=candidate.component,
        location=candidate.location_hint,
        defect_type=defect_type,
        deru=deru,
        method=InspectionMethod.AI_DRONE,
        requires_ndt=requires_ndt,
        ndt_method=ndt_method,
        image_path=candidate.annotated_image_path,
        ai_confidence=candidate.confidence,
        human_reviewed=True,
        note=note,
    )


# ---------------------------------------------------------------------------
# 5. 檢測週期規則引擎(依交通部規範精神實作)
# ---------------------------------------------------------------------------
class TaiwanInspectionScheduler:
    """
    依交通部規範推算橋梁下次檢測到期日:
      - 定期檢測: 每 2 年(24 個月)一次
      - 日常巡查: 每年 4 次(約每 3 個月一次)
      - 詳細檢測: 定期檢測發現疑慮構件後,建議 1 年內完成
      - 特別檢測: 事件驅動(地震、颱風、事故後),無固定週期
      - 若構件 U(急迫性) >= 3,定期檢測週期須提前縮短,不受一般週期限制
    """

    DEFAULT_INTERVALS_MONTHS = {
        TaiwanInspectionType.ROUTINE: 24,
        TaiwanInspectionType.DAILY_PATROL: 3,
        TaiwanInspectionType.DETAILED: 12,
        TaiwanInspectionType.SPECIAL: 0,   # 事件驅動,無固定週期
    }

    @classmethod
    def next_due_date(cls, base_date: date, inspection_type: TaiwanInspectionType,
                       max_urgency: int = 0) -> date:
        months = cls.DEFAULT_INTERVALS_MONTHS.get(inspection_type, 24)

        # 若有構件 U>=3,依規定須加速處置,定期檢測週期強制縮短為 12 個月內複查
        if inspection_type == TaiwanInspectionType.ROUTINE and max_urgency >= 3:
            months = 12
        if inspection_type == TaiwanInspectionType.ROUTINE and max_urgency >= 4:
            months = 0  # U=4 即刻通報緊急處置,不待排程

        if months == 0:
            return base_date

        year = base_date.year + (base_date.month - 1 + months) // 12
        month = (base_date.month - 1 + months) % 12 + 1
        day = min(base_date.day, 28)  # 避免月底日期溢位問題
        return date(year, month, day)


# ---------------------------------------------------------------------------
# 6. 使用範例
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    record = BridgeInspectionRecord(
        bridge_id="TW-BR-00123",
        inspection_type=TaiwanInspectionType.ROUTINE,
        inspection_date=date(2026, 8, 9),
        inspector="王小明",
    )

    # ------------------------------------------------------------------
    # 範例A-1: 批次處理整個資料夾的無人機照片,依風險排序找出優先複核對象
    # ------------------------------------------------------------------
    try:
        candidates = batch_analyze_drone_folder(
            folder_path="drone_photos",   # 換成無人機這次飛完後的照片資料夾
            component=BridgeComponent.SUPERSTRUCTURE,
        )
        print_review_priority_list(candidates, top_n=5)
    except NotADirectoryError as e:
        print(f"[批次處理略過: {e}]")
        candidates = []
    print()

    # ------------------------------------------------------------------
    # 範例A-2: 單張影像 AI 初篩 -> 人工複核 -> 正式寫入缺陷紀錄
    # (若無實際無人機影像/未安裝 opencv-python,這段會被 try/except 跳過)
    # ------------------------------------------------------------------
    try:
        candidate = analyze_drone_image_for_cracks(
            image_path="drone_photos/pier2_south_001.jpg",  # 換成無人機實際拍攝的照片路徑
            component=BridgeComponent.SUPERSTRUCTURE,
            location_hint="主梁跨中下翼緣,南側第2跨(無人機航點 WP-12)",
        )
        # AI 只做初篩,DERU 仍須由檢測人員現場判斷或依標註影像覆核後輸入
        crack_defect = confirm_defect_from_drone_candidate(
            candidate=candidate,
            defect_type="疑似疲勞裂縫",
            deru=DERURating(degree=3, extent=2, relevancy=3, urgency=3),
            requires_ndt=True,
            ndt_method="染色滲透探傷 (Dye Penetrant Testing)",
            reviewer_note="經檢測人員複核,確認為疑似疲勞裂縫,建議安排NDT確認範圍。",
        )
    except (FileNotFoundError, ImportError) as e:
        # 沒有照片或未安裝 opencv 時,退回手動輸入(示範用)
        print(f"[無人機影像分析略過: {e}]\n")
        crack_defect = DefectRecord(
            component=BridgeComponent.SUPERSTRUCTURE,
            location="主梁跨中下翼緣,南側第2跨",
            defect_type="疑似疲勞裂縫",
            deru=DERURating(degree=3, extent=2, relevancy=3, urgency=3),
            method=InspectionMethod.AI_DRONE,
            requires_ndt=True,
            ndt_method="染色滲透探傷 (Dye Penetrant Testing)",
            note="AI無人機影像辨識初判疑似裂縫,建議安排染色滲透檢測確認裂縫範圍",
        )
    record.add_defect(crack_defect)

    # 範例B: 傳統人工目視發現的缺陷,直接手動輸入
    corrosion_defect = DefectRecord(
        component=BridgeComponent.SUBSTRUCTURE,
        location="橋墩P2水位變動區",
        defect_type="混凝土保護層剝落、鋼筋外露",
        deru=DERURating(degree=2, extent=1, relevancy=2, urgency=2),
        requires_ndt=False,
        note="建議納入年度修繕計畫,持續觀察是否擴大",
    )
    record.add_defect(corrosion_defect)

    print(record.summary())
    print()

    print("--- 優先修繕清單 (U >= 3) ---")
    for d in record.priority_repair_defects():
        print(f"  [{d.component.name}] {d.location}: {d.defect_type} -> {d.deru.repair_deadline_note}")
    print()

    next_due = TaiwanInspectionScheduler.next_due_date(
        base_date=record.inspection_date,
        inspection_type=TaiwanInspectionType.ROUTINE,
        max_urgency=record.max_urgency(),
    )
    print(f"下次定期檢測到期日(已因最高 U={record.max_urgency()} 縮短週期): {next_due}")