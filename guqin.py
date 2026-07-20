#!/usr/bin/env python3  # 告訴 macOS/Linux：用 python3 來執行這個檔案。
"""古琴減字譜機器學習教學版。"""  # 三引號字串是檔案說明，也叫 module docstring。

from __future__ import annotations  # 讓型別註解可以晚一點再解析，寫起來比較彈性。

import argparse  # argparse 用來做命令列指令，例如 train、predict、review。
import csv  # csv 用來讀寫 labels.csv 與 mapping.csv。
import math  # math 提供 floor 等數學工具，這裡用來切訓練/驗證資料。
import random  # random 用來打亂訓練資料，避免模型只記住資料順序。
import sys  # sys.stdout 代表終端機輸出，predict 會把 CSV 印到終端機。
from dataclasses import dataclass  # dataclass 用來建立簡潔的資料容器。
from pathlib import Path  # Path 比字串路徑更安全，也比較容易處理資料夾與副檔名。
from typing import Any, Iterable  # Any/Iterable 是型別提示，幫助人和編輯器理解資料形狀。


LABEL_FIELDS = ("image_path", "mode", "string", "hui", "technique", "jianpu")  # 標註檔必須有的欄位。
MAPPING_FIELDS = ("mode", "string", "hui", "technique", "jianpu", "pitch")  # 對照表必須有的欄位。
TARGET_FIELDS = ("mode", "string", "hui", "technique", "jianpu")  # 模型要從圖片學會預測的目標。
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}  # 可接受的圖片副檔名。


def require_ml_dependencies() -> tuple[Any, Any, Any, Any, Any]:  # 定義函式：需要機器學習套件時才載入。
    """載入訓練需要的外部套件；缺少時給使用者清楚的安裝指令。"""  # 說明函式用途。
    try:  # try 代表先嘗試執行，若失敗就交給 except。
        import numpy as np  # numpy 用來把圖片像素轉成數字矩陣。
        from PIL import Image, ImageOps  # pillow 的 Image 讀圖片，ImageOps 做圖片增強。
        import torch  # torch 是 PyTorch，負責張量、模型與訓練。
        import torch.nn as nn  # torch.nn 提供神經網路層，例如 Conv2d、Linear。
    except ModuleNotFoundError as exc:  # 如果缺少套件，就抓住錯誤並命名為 exc。
        missing = exc.name or "machine-learning dependency"  # exc.name 會告訴我們缺哪個套件。
        raise SystemExit(  # SystemExit 會友善結束程式，不顯示一長串嚇人的錯誤。
            f"缺少套件：{missing}\n"  # f-string 可以把變數 missing 放進字串。
            "請先安裝：\n"  # 多個字串放在一起，Python 會自動接起來。
            "  python3 -m pip install torch pillow numpy"  # 這是訓練需要的安裝指令。
        ) from exc  # from exc 保留原始錯誤來源，方便除錯。
    return np, Image, ImageOps, torch, nn  # 回傳這些套件，讓其他函式使用。


def read_csv_rows(path: Path, required_fields: Iterable[str]) -> list[dict[str, str]]:  # 讀 CSV 並檢查欄位。
    """讀取 CSV，並確認需要的欄位都存在。"""  # 函式文件字串，說明目的。
    if not path.exists():  # 如果檔案不存在，就不能繼續讀。
        raise SystemExit(f"找不到 CSV 檔案：{path}")  # 直接用清楚訊息停止。

    with path.open("r", encoding="utf-8-sig", newline="") as file:  # 用 UTF-8 讀檔，utf-8-sig 可處理 Excel BOM。
        reader = csv.DictReader(file)  # DictReader 會把每一列變成 dict，例如 {"mode": "黃鐘正調"}。
        fieldnames = reader.fieldnames or []  # 如果 CSV 沒表頭，fieldnames 可能是 None，所以用空 list 代替。
        missing = [field for field in required_fields if field not in fieldnames]  # 找出缺少的必要欄位。
        if missing:  # 如果 missing 不是空的，代表 CSV 格式不正確。
            raise SystemExit(f"{path} 缺少欄位：{', '.join(missing)}")  # join 把欄位名稱串成好讀文字。
        rows = []  # 準備一個 list，用來裝清理過的每一列。
        for row in reader:  # 逐列讀取 CSV。
            clean_row = {key: (value or "").strip() for key, value in row.items()}  # 去掉前後空白，None 改成空字串。
            rows.append(clean_row)  # 把整理好的資料列放進 rows。
        return rows  # 回傳所有資料列。


def write_csv_if_missing(path: Path, fields: Iterable[str], rows: list[dict[str, str]]) -> None:  # 建立範例 CSV。
    """如果 CSV 還不存在，就建立一份有表頭和範例列的檔案。"""  # 說明函式目的。
    if path.exists():  # 如果檔案已存在，代表使用者可能已經填資料。
        return  # 不覆蓋舊檔，避免把使用者資料洗掉。
    path.parent.mkdir(parents=True, exist_ok=True)  # 建立上層資料夾；exist_ok=True 表示已存在也不報錯。
    with path.open("w", encoding="utf-8", newline="") as file:  # 用寫入模式開檔。
        writer = csv.DictWriter(file, fieldnames=list(fields))  # DictWriter 用 dict 寫 CSV。
        writer.writeheader()  # 先寫表頭。
        writer.writerows(rows)  # 再寫範例資料。


def append_label(path: Path, row: dict[str, str]) -> None:  # 把人工確認後的標註加到 labels.csv。
    """把一筆標註追加到 labels.csv，給下一次訓練使用。"""  # 函式用途。
    path.parent.mkdir(parents=True, exist_ok=True)  # 確保資料夾存在。
    file_exists = path.exists()  # 記住檔案原本是否存在。
    with path.open("a", encoding="utf-8", newline="") as file:  # "a" 是 append，會加在檔案最後。
        writer = csv.DictWriter(file, fieldnames=list(LABEL_FIELDS))  # 設定固定欄位順序。
        if not file_exists:  # 如果是新檔案，就需要先寫表頭。
            writer.writeheader()  # 寫入 CSV 表頭。
        writer.writerow({field: row.get(field, "") for field in LABEL_FIELDS})  # 只寫我們需要的欄位。


def collect_image_paths(paths: Iterable[str]) -> list[Path]:  # 收集圖片路徑，可接受檔案或資料夾。
    """從使用者輸入的檔案/資料夾中找出所有圖片。"""  # 函式用途。
    found: list[Path] = []  # 建立空 list，準備存圖片路徑。
    for raw_path in paths:  # 使用者可能輸入多個路徑，所以逐一處理。
        path = Path(raw_path)  # 把字串轉成 Path 物件。
        if path.is_dir():  # 如果是資料夾，就要找資料夾內的圖片。
            children = path.rglob("*")  # rglob("*") 會遞迴找出資料夾下所有檔案。
            images = [child for child in children if child.suffix.lower() in IMAGE_EXTENSIONS]  # 只保留圖片副檔名。
            found.extend(sorted(images))  # sorted 讓順序固定，extend 把多個路徑加進 found。
        elif path.suffix.lower() in IMAGE_EXTENSIONS:  # 如果是單張圖片，就直接接受。
            found.append(path)  # 把圖片路徑放進 found。
    return found  # 回傳收集到的圖片。


def normalize_label(value: str) -> str:  # 把標籤轉成模型可用文字。
    """把空白標籤轉成特殊字，避免模型類別裡出現真正的空字串。"""  # 說明原因。
    return value.strip() or "<blank>"  # strip 去空白；如果結果是空字串，就用 <blank>。


def denormalize_label(value: str) -> str:  # 把模型內部標籤轉回人類看的文字。
    """把模型內部的 <blank> 還原成空字串。"""  # 說明用途。
    return "" if value == "<blank>" else value  # 三元運算式：條件成立回傳前者，否則回傳後者。


@dataclass(frozen=True)  # dataclass 自動建立 __init__；frozen=True 表示建立後不能改。
class MappingResult:  # 定義對照表查詢結果。
    jianpu: str  # jianpu 儲存簡譜，例如 "6"。
    pitch: str  # pitch 儲存國際音高或唱名，例如 "A/la"。


class NoteMapping:  # 這個類別負責「指法/弦/徽/調式」到「簡譜」的查表。
    """把 mapping.csv 變成快速查詢表。"""  # 類別用途。

    def __init__(self, rows: list[dict[str, str]]):  # 建立 NoteMapping 時，需要傳入 CSV rows。
        self.exact: dict[tuple[str, str, str, str], MappingResult] = {}  # exact 用四個條件精準查詢。
        self.no_technique: dict[tuple[str, str, str], MappingResult] = {}  # no_technique 忽略指法查詢。
        for row in rows:  # 每一列 mapping.csv 都是一個音高規則。
            mode = row.get("mode", "").strip()  # 讀調式，例如 黃鐘正調。
            string = row.get("string", "").strip()  # 讀弦，例如 四。
            hui = row.get("hui", "").strip()  # 讀徽位，例如 九。
            technique = row.get("technique", "").strip()  # 讀指法，例如 勾。
            jianpu = row.get("jianpu", "").strip()  # 讀簡譜，例如 6。
            pitch = row.get("pitch", "").strip()  # 讀音高，例如 A/la。
            if not jianpu:  # 如果沒有簡譜，這列就不能提供答案。
                continue  # 跳過這列，繼續處理下一列。
            result = MappingResult(jianpu=jianpu, pitch=pitch)  # 包成 MappingResult，讓資料更有結構。
            self.exact[(mode, string, hui, technique)] = result  # 建立完整 key 的查表資料。
            self.no_technique[(mode, string, hui)] = result  # 也建立忽略指法的備用查表資料。

    def lookup(self, mode: str, string: str, hui: str, technique: str) -> MappingResult | None:  # 查詢簡譜。
        exact_key = (mode, string, hui, technique)  # 先組成完整查詢 key。
        if exact_key in self.exact:  # 如果完整條件查得到，就用完整答案。
            return self.exact[exact_key]  # 回傳精準查詢結果。
        loose_key = (mode, string, hui)  # 如果查不到，就改用不含指法的 key。
        return self.no_technique.get(loose_key)  # dict.get 查不到會回傳 None，不會丟錯。


class Vocab:  # Vocab 負責把文字標籤轉成數字類別。
    """模型只能學數字類別，所以要把中文字標籤編碼。"""  # 類別用途。

    def __init__(self, values_by_field: dict[str, list[str]]):  # 建立 Vocab，傳入每個欄位的所有可能值。
        self.values_by_field = values_by_field  # 保存 index -> 文字 的清單。
        self.index_by_field = {}  # 準備保存 文字 -> index 的查表。
        for field, values in values_by_field.items():  # 每個欄位都要建立自己的字典。
            self.index_by_field[field] = {value: index for index, value in enumerate(values)}  # enumerate 產生編號。

    @classmethod  # classmethod 代表這個函式屬於類別，不需要先建立物件。
    def from_rows(cls, rows: list[dict[str, str]]) -> "Vocab":  # 從 labels.csv 建立 Vocab。
        values_by_field: dict[str, list[str]] = {}  # 準備收集每個欄位有哪些類別。
        for field in TARGET_FIELDS:  # 模型要預測的每個欄位都處理一次。
            values = {normalize_label(row.get(field, "")) for row in rows}  # set comprehension 會去除重複標籤。
            sorted_values = sorted(values)  # 排序讓類別編號固定，重跑訓練比較穩定。
            if not sorted_values:  # 如果沒有任何類別，模型不能訓練。
                raise SystemExit(f"欄位沒有標籤：{field}")  # 顯示哪個欄位出問題。
            values_by_field[field] = sorted_values  # 把這個欄位的類別清單存起來。
        return cls(values_by_field)  # cls(...) 等於建立 Vocab 物件。

    def encode(self, field: str, value: str) -> int:  # 把文字標籤轉成整數。
        key = normalize_label(value)  # 先標準化，讓空字串可被編碼。
        return self.index_by_field[field][key]  # 用字典查出類別編號。

    def decode(self, field: str, index: int) -> str:  # 把整數轉回文字標籤。
        value = self.values_by_field[field][index]  # 用 index 從清單取出標籤。
        return denormalize_label(value)  # 把 <blank> 還原成人類習慣的空字串。

    def sizes(self) -> dict[str, int]:  # 回傳每個預測欄位有幾種類別。
        return {field: len(values) for field, values in self.values_by_field.items()}  # len 是該欄位分類數。


def load_image_tensor(path: Path, image_size: int) -> Any:  # 把圖片檔案轉成模型可吃的張量。
    """讀圖片、轉灰階、置中縮放、轉成 PyTorch tensor。"""  # 說明圖片前處理。
    np, Image, ImageOps, torch, _ = require_ml_dependencies()  # 載入需要的套件；_ 表示這個回傳值不用。
    with Image.open(path) as image:  # 開啟圖片；with 會自動關檔。
        image = image.convert("L")  # "L" 是灰階；減字譜重點是筆畫形狀，不需要顏色。
        image = ImageOps.autocontrast(image)  # 自動拉高對比，讓黑色筆畫更清楚。
        image.thumbnail((image_size, image_size))  # 保持比例縮小到指定尺寸內。
        canvas = Image.new("L", (image_size, image_size), 255)  # 建立白色正方形畫布。
        left = (image_size - image.width) // 2  # 算出水平置中的左邊距。
        top = (image_size - image.height) // 2  # 算出垂直置中的上邊距。
        canvas.paste(image, (left, top))  # 把縮好的圖片貼到白色畫布中央。
        array = np.asarray(canvas, dtype=np.float32) / 255.0  # 像素轉成 0~1 的浮點數矩陣。
        array = 1.0 - array  # 反轉黑白，讓黑色筆畫變成接近 1，背景變成接近 0。
        tensor = torch.from_numpy(array).unsqueeze(0)  # 轉 tensor，unsqueeze(0) 增加 channel 維度。
        return tensor  # 回傳形狀約為 [1, image_size, image_size] 的張量。


def make_model(output_sizes: dict[str, int]) -> Any:  # 建立神經網路模型。
    """建立一個小型 CNN，同時預測調式、弦、徽位、指法、簡譜。"""  # 說明模型架構。
    _, _, _, _, nn = require_ml_dependencies()  # 这里只需要 torch.nn。

    class GuqinCNN(nn.Module):  # 定義內部模型類別，繼承 PyTorch 的 nn.Module。
        def __init__(self) -> None:  # 初始化模型層。
            super().__init__()  # 呼叫 nn.Module 的初始化，這是 PyTorch 必要寫法。
            self.features = nn.Sequential(  # Sequential 代表把多層依序串起來。
                nn.Conv2d(1, 24, kernel_size=3, padding=1),  # 卷積層：從 1 個灰階通道抽 24 種筆畫特徵。
                nn.ReLU(),  # ReLU 加入非線性，讓模型能學複雜形狀。
                nn.MaxPool2d(2),  # 池化層縮小圖片，保留重要特徵並降低計算量。
                nn.Conv2d(24, 48, kernel_size=3, padding=1),  # 第二層卷積，學更複雜的局部結構。
                nn.ReLU(),  # 再次加入非線性。
                nn.MaxPool2d(2),  # 再縮小一次特徵圖。
                nn.Conv2d(48, 96, kernel_size=3, padding=1),  # 第三層卷積，整理高階字形特徵。
                nn.ReLU(),  # 非線性啟動。
                nn.AdaptiveAvgPool2d((1, 1)),  # 把任意大小特徵壓成每通道 1 個數字。
            )  # features 到這裡結束。
            self.heads = nn.ModuleDict()  # ModuleDict 存多個輸出頭，PyTorch 才會追蹤參數。
            for field, size in output_sizes.items():  # 每個預測欄位需要一個分類器。
                self.heads[field] = nn.Linear(96, size)  # Linear 把 96 個特徵轉成該欄位的類別分數。

        def forward(self, images: Any) -> dict[str, Any]:  # forward 定義圖片如何通過模型。
            features = self.features(images).flatten(1)  # 抽特徵後展平成 [batch, 96]。
            outputs = {field: head(features) for field, head in self.heads.items()}  # 每個 head 都輸出分類分數。
            return outputs  # 回傳 dict，例如 {"hui": logits, "string": logits}。

    return GuqinCNN()  # 建立並回傳模型物件。


def make_batches(rows: list[dict[str, str]], batch_size: int) -> Iterable[list[dict[str, str]]]:  # 把資料切成批次。
    """把很多列資料切成一批一批，訓練時比較有效率。"""  # 說明原因。
    for start in range(0, len(rows), batch_size):  # range 每次跳 batch_size。
        end = start + batch_size  # 算出這批的結束位置。
        yield rows[start:end]  # yield 逐批產生資料，不一次複製全部。


def evaluate(model: Any, rows: list[dict[str, str]], vocab: Vocab, image_size: int) -> dict[str, float]:  # 驗證模型準確率。
    """用驗證資料估計模型目前每個欄位的準確率。"""  # 函式用途。
    if not rows:  # 如果沒有驗證資料，就不能算準確率。
        return {field: 0.0 for field in TARGET_FIELDS}  # 回傳 0，避免除以 0。
    _, _, _, torch, _ = require_ml_dependencies()  # 評估時需要 torch.no_grad。
    correct = {field: 0 for field in TARGET_FIELDS}  # 記錄每個欄位答對幾題。
    model.eval()  # 切到評估模式，關閉訓練專用行為。
    with torch.no_grad():  # 評估不需要反向傳播，可節省記憶體。
        for row in rows:  # 一張一張驗證。
            image = load_image_tensor(Path(row["image_path"]), image_size).unsqueeze(0)  # 增加 batch 維度。
            logits = model(image)  # 模型輸出每個欄位的分類分數。
            for field in TARGET_FIELDS:  # 每個欄位分別計算正確與否。
                predicted_index = int(logits[field].argmax(dim=1).item())  # argmax 找分數最高的類別。
                true_index = vocab.encode(field, row.get(field, ""))  # 把正確標籤轉成類別編號。
                if predicted_index == true_index:  # 如果預測類別等於正確類別。
                    correct[field] += 1  # 該欄位答對數加一。
    return {field: correct[field] / len(rows) for field in TARGET_FIELDS}  # 答對數除以總數就是準確率。


def train(args: argparse.Namespace) -> None:  # train 子指令的主函式。
    """讀 labels.csv，訓練模型，並存成 .pt 檔。"""  # 說明用途。
    _, _, _, torch, nn = require_ml_dependencies()  # 訓練需要 torch 和 nn。
    rows = read_csv_rows(Path(args.labels), LABEL_FIELDS)  # 讀取訓練標註。
    rows = [row for row in rows if row.get("image_path")]  # 移除沒有圖片路徑的空列。
    if len(rows) < 2:  # 至少要兩筆，才能切訓練與驗證。
        raise SystemExit("至少需要 2 張已標註圖片才能訓練。")  # 給出明確錯誤。
    for row in rows:  # 檢查每張圖片是否真的存在。
        if not Path(row["image_path"]).exists():  # 如果圖片路徑不存在。
            raise SystemExit(f"找不到訓練圖片：{row['image_path']}")  # 停止並指出問題檔案。

    random.seed(args.seed)  # 固定亂數種子，讓切分結果較可重現。
    random.shuffle(rows)  # 打亂資料，避免同類圖片集中在一起。
    split = math.floor(len(rows) * (1.0 - args.val_ratio))  # 算訓練資料數量。
    split = max(1, min(len(rows) - 1, split))  # 保證訓練集和驗證集都至少有一筆。
    train_rows = rows[:split]  # 前半段當訓練資料。
    val_rows = rows[split:]  # 後半段當驗證資料。

    vocab = Vocab.from_rows(rows)  # 從所有標註建立文字/數字對照。
    model = make_model(vocab.sizes())  # 依照每個欄位的類別數建立模型。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  # AdamW 負責更新權重。
    loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss 是多類別分類常用損失函式。

    for epoch in range(1, args.epochs + 1):  # epoch 代表完整看過訓練資料一次。
        model.train()  # 切到訓練模式。
        random.shuffle(train_rows)  # 每個 epoch 都重排資料，讓學習更穩。
        total_loss = 0.0  # 累積整個 epoch 的損失。
        for batch in make_batches(train_rows, args.batch_size):  # 每次取一批資料訓練。
            images = [load_image_tensor(Path(row["image_path"]), args.image_size) for row in batch]  # 讀入這批圖片。
            image_tensor = torch.stack(images)  # stack 把多張 [1,H,W] 疊成 [batch,1,H,W]。
            targets = {}  # 準備每個欄位的正確答案張量。
            for field in TARGET_FIELDS:  # 每個預測欄位都需要答案。
                labels = [vocab.encode(field, row.get(field, "")) for row in batch]  # 文字答案轉數字。
                targets[field] = torch.tensor(labels)  # 轉成 PyTorch tensor。

            logits = model(image_tensor)  # 把圖片送進模型，得到每個欄位的分類分數。
            losses = [loss_fn(logits[field], targets[field]) for field in TARGET_FIELDS]  # 每個欄位各算一個 loss。
            loss = sum(losses)  # 多任務模型把各欄位 loss 加總，一起學。
            optimizer.zero_grad()  # 清掉上一批留下的梯度。
            loss.backward()  # 反向傳播，計算每個參數該怎麼改。
            optimizer.step()  # 根據梯度更新模型參數。
            total_loss += float(loss.item()) * len(batch)  # 累積 loss，乘 batch 大小方便算平均。

        metrics = evaluate(model, val_rows, vocab, args.image_size)  # 每個 epoch 後驗證一次。
        average_loss = total_loss / len(train_rows)  # 算平均訓練 loss。
        metric_text = " ".join(f"{field}_acc={score:.2f}" for field, score in metrics.items())  # 格式化準確率。
        print(f"epoch={epoch:03d} loss={average_loss:.4f} {metric_text}")  # 印出訓練進度。

    model_path = Path(args.model)  # 把模型輸出路徑轉成 Path。
    model_path.parent.mkdir(parents=True, exist_ok=True)  # 確保 models/ 資料夾存在。
    torch.save(  # torch.save 會把模型狀態存成 .pt 檔。
        {  # 存成 dict，之後載入時才知道模型權重、類別表、圖片大小。
            "model_state": model.state_dict(),  # state_dict 是模型所有權重。
            "vocab": vocab.values_by_field,  # vocab 保存類別編號與文字的關係。
            "image_size": args.image_size,  # 預測時必須用同樣圖片大小。
        },  # dict 結束。
        model_path,  # 存檔位置。
    )  # torch.save 呼叫結束。
    print(f"模型已存檔：{model_path}")  # 告訴使用者模型位置。


def load_checkpoint(model_path: Path) -> tuple[Any, Vocab, int]:  # 載入已訓練模型。
    """讀取 .pt 模型檔，重建模型與 vocab。"""  # 函式用途。
    _, _, _, torch, _ = require_ml_dependencies()  # 載入 torch。
    checkpoint = torch.load(model_path, map_location="cpu")  # map_location="cpu" 表示沒有 GPU 也能讀。
    vocab = Vocab(checkpoint["vocab"])  # 用存檔內的 vocab 重建文字/數字對照。
    image_size = int(checkpoint["image_size"])  # 讀取訓練時使用的圖片尺寸。
    model = make_model(vocab.sizes())  # 用相同輸出大小重建模型架構。
    model.load_state_dict(checkpoint["model_state"])  # 把訓練好的權重放回模型。
    model.eval()  # 預測時使用評估模式。
    return model, vocab, image_size  # 回傳模型、vocab、圖片大小。


def predict_one(model: Any, vocab: Vocab, image_size: int, path: Path) -> dict[str, str]:  # 預測單張圖片。
    """對單張減字譜圖片預測調式、弦、徽位、指法、簡譜。"""  # 函式用途。
    _, _, _, torch, _ = require_ml_dependencies()  # 預測需要 torch.no_grad。
    with torch.no_grad():  # 預測不需要訓練梯度。
        image = load_image_tensor(path, image_size).unsqueeze(0)  # 讀圖並加 batch 維度。
        logits = model(image)  # 取得每個欄位的類別分數。
        result = {}  # 準備存預測結果。
        for field in TARGET_FIELDS:  # 逐欄位解碼。
            index = int(logits[field].argmax(dim=1).item())  # 找分數最高的類別 index。
            result[field] = vocab.decode(field, index)  # 把 index 轉回文字。
        return result  # 回傳像 {"string": "四", "hui": "九"} 的 dict。


def predict(args: argparse.Namespace) -> None:  # predict 子指令的主函式。
    """預測圖片，並用 mapping.csv 把結果修正/映射成簡譜與音高。"""  # 函式用途。
    model, vocab, image_size = load_checkpoint(Path(args.model))  # 載入模型。
    mapping_rows = read_csv_rows(Path(args.mapping), MAPPING_FIELDS)  # 讀取對照表。
    mapping = NoteMapping(mapping_rows)  # 建立可查詢的 NoteMapping。
    paths = collect_image_paths(args.paths)  # 收集要預測的圖片。
    if not paths:  # 如果找不到圖片。
        raise SystemExit("沒有找到圖片。")  # 提醒使用者。

    fieldnames = ["image_path", "mode", "string", "hui", "technique", "jianpu", "pitch"]  # 預測輸出的 CSV 欄位。
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)  # 寫到 stdout，也就是終端機。
    writer.writeheader()  # 先印 CSV 表頭。
    for path in paths:  # 每張圖片逐一預測。
        row = predict_one(model, vocab, image_size, path)  # 模型先預測五個欄位。
        mapped = mapping.lookup(row["mode"], row["string"], row["hui"], row["technique"])  # 用音樂規則查簡譜。
        row["image_path"] = str(path)  # 加入圖片路徑。
        row["pitch"] = ""  # 預設音高是空字串，避免欄位不存在。
        if mapped:  # 如果對照表查得到。
            row["jianpu"] = mapped.jianpu  # 用對照表的簡譜覆蓋模型簡譜，因為規則更可靠。
            row["pitch"] = mapped.pitch  # 補上音高。
        writer.writerow(row)  # 印出這張圖片的預測結果。


def review(args: argparse.Namespace) -> None:  # review 子指令的主函式。
    """讓人逐張確認模型預測，接受或修正後追加進 labels.csv。"""  # 函式用途。
    model, vocab, image_size = load_checkpoint(Path(args.model))  # 載入模型。
    mapping = NoteMapping(read_csv_rows(Path(args.mapping), MAPPING_FIELDS))  # 載入音高對照表。
    paths = collect_image_paths(args.paths)  # 找出要抽查的圖片。
    if not paths:  # 如果沒有圖片。
        raise SystemExit("沒有找到圖片。")  # 停止並提醒。

    for path in paths:  # 一張一張讓人看。
        predicted = predict_one(model, vocab, image_size, path)  # 先讓模型預測。
        mapped = mapping.lookup(predicted["mode"], predicted["string"], predicted["hui"], predicted["technique"])  # 查規則。
        jianpu = mapped.jianpu if mapped else predicted["jianpu"]  # 有規則就用規則，沒有就用模型答案。
        print(f"\n圖片：{path}")  # 顯示目前圖片路徑。
        print(  # 印出模型預測，方便人工判斷。
            f"預測：調={predicted['mode']} 弦={predicted['string']} "
            f"徽={predicted['hui']} 指法={predicted['technique']} 簡譜={jianpu}"
        )  # print 結束。
        answer = input("接受嗎？[Y/n/edit/skip] ").strip().lower()  # 讀使用者輸入並轉小寫。
        if answer in {"", "y", "yes"}:  # 空白或 yes 代表接受。
            row = dict(predicted)  # 複製預測結果當成新標註。
            row["jianpu"] = jianpu  # 確保簡譜使用規則修正後的值。
        elif answer in {"s", "skip"}:  # skip 代表這張先不處理。
            continue  # 跳到下一張。
        else:  # 其他輸入代表要手動修正。
            row = {}  # 準備人工標註 dict。
            for field in TARGET_FIELDS:  # 每個欄位都問一次。
                default = jianpu if field == "jianpu" else predicted[field]  # 簡譜欄位預設用查表後的值。
                value = input(f"{field} [{default}]: ").strip()  # 顯示預設值，讓使用者可直接按 Enter。
                row[field] = value or default  # 如果沒輸入，就使用預設值。
        row["image_path"] = str(path)  # 補上圖片路徑。
        append_label(Path(args.labels), row)  # 把確認後的答案追加到 labels.csv。
        print(f"已加入標註：{args.labels}")  # 告訴使用者已保存。


def init_project(args: argparse.Namespace) -> None:  # init 子指令的主函式。
    """建立資料夾與範例 CSV，讓使用者知道資料格式。"""  # 函式用途。
    data_dir = Path(args.data_dir)  # 把資料夾參數轉成 Path。
    (data_dir / "images").mkdir(parents=True, exist_ok=True)  # 建立 data/images。
    Path("models").mkdir(exist_ok=True)  # 建立 models 資料夾。
    write_csv_if_missing(  # 如果 labels.csv 不存在，就建立範例。
        data_dir / "labels.csv",  # 標註檔路徑。
        LABEL_FIELDS,  # 標註檔欄位。
        [  # 範例資料 list 開始。
            {  # 一筆範例標註開始。
                "image_path": "data/images/example.png",  # 圖片位置。
                "mode": "黃鐘正調",  # 調式。
                "string": "四",  # 第四弦。
                "hui": "九",  # 第九徽。
                "technique": "勾",  # 指法是勾。
                "jianpu": "6",  # 簡譜 la。
            }  # 一筆範例標註結束。
        ],  # 範例資料 list 結束。
    )  # write_csv_if_missing 結束。
    write_csv_if_missing(  # 如果 mapping.csv 不存在，就建立範例。
        data_dir / "mapping.csv",  # 對照表路徑。
        MAPPING_FIELDS,  # 對照表欄位。
        [  # 範例對照資料開始。
            {"mode": "黃鐘正調", "string": "四", "hui": "九", "technique": "勾", "jianpu": "6", "pitch": "A/la"},  # 範例：四弦九徽勾為 la。
            {"mode": "黃鐘正調", "string": "五", "hui": "七", "technique": "", "jianpu": "6", "pitch": "A/la"},  # 範例：五弦七徽也是 la。
        ],  # 範例對照資料結束。
    )  # write_csv_if_missing 結束。
    print(f"已建立資料夾與 CSV 範例：{data_dir}")  # 顯示初始化完成。
    print("請把圖片放進 data/images，並編輯 data/labels.csv 與 data/mapping.csv。")  # 下一步提示。


def build_parser() -> argparse.ArgumentParser:  # 建立命令列解析器。
    """定義這個程式有哪些子指令與參數。"""  # 函式用途。
    parser = argparse.ArgumentParser(description="古琴減字譜影像模型訓練工具。")  # 建立最外層 parser。
    subparsers = parser.add_subparsers(dest="command", required=True)  # 子指令，例如 init/train/predict/review。

    init_cmd = subparsers.add_parser("init", help="建立資料夾與 CSV 範例")  # 新增 init 子指令。
    init_cmd.add_argument("--data-dir", default="data")  # init 可指定資料夾，預設 data。
    init_cmd.set_defaults(func=init_project)  # 執行 init 時呼叫 init_project。

    train_cmd = subparsers.add_parser("train", help="訓練模型")  # 新增 train 子指令。
    train_cmd.add_argument("--labels", default="data/labels.csv")  # labels.csv 位置。
    train_cmd.add_argument("--model", default="models/guqin_jianzipu.pt")  # 模型輸出位置。
    train_cmd.add_argument("--epochs", type=int, default=40)  # 訓練回合數。
    train_cmd.add_argument("--batch-size", type=int, default=16)  # 每批圖片數。
    train_cmd.add_argument("--image-size", type=int, default=96)  # 圖片正方形尺寸。
    train_cmd.add_argument("--lr", type=float, default=1e-3)  # learning rate，控制每次更新幅度。
    train_cmd.add_argument("--val-ratio", type=float, default=0.2)  # 驗證資料比例。
    train_cmd.add_argument("--seed", type=int, default=7)  # 亂數種子。
    train_cmd.set_defaults(func=train)  # 執行 train 時呼叫 train 函式。

    predict_cmd = subparsers.add_parser("predict", help="預測圖片")  # 新增 predict 子指令。
    predict_cmd.add_argument("paths", nargs="+")  # 一個或多個圖片/資料夾路徑。
    predict_cmd.add_argument("--model", default="models/guqin_jianzipu.pt")  # 要載入的模型。
    predict_cmd.add_argument("--mapping", default="data/mapping.csv")  # 對照表路徑。
    predict_cmd.set_defaults(func=predict)  # 執行 predict 時呼叫 predict。

    review_cmd = subparsers.add_parser("review", help="人工抽查並加入新標註")  # 新增 review 子指令。
    review_cmd.add_argument("paths", nargs="+")  # 要抽查的一個或多個圖片/資料夾。
    review_cmd.add_argument("--model", default="models/guqin_jianzipu.pt")  # 要載入的模型。
    review_cmd.add_argument("--mapping", default="data/mapping.csv")  # 對照表路徑。
    review_cmd.add_argument("--labels", default="data/labels.csv")  # 修正後要寫入的標註檔。
    review_cmd.set_defaults(func=review)  # 執行 review 時呼叫 review。

    return parser  # 回傳設定好的 parser。


def main(argv: list[str] | None = None) -> int:  # 程式進入點；argv 可測試時手動傳入。
    """解析命令列，並執行對應子指令。"""  # 函式用途。
    parser = build_parser()  # 建立命令列 parser。
    args = parser.parse_args(argv)  # 解析使用者輸入的指令與參數。
    args.func(args)  # 呼叫子指令綁定的函式，例如 train(args)。
    return 0  # 回傳 0 表示程式正常結束。


if __name__ == "__main__":  # 只有直接執行 python3 guqin.py 時才成立。
    raise SystemExit(main())  # 呼叫 main，並把回傳碼交給作業系統。
