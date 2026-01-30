import json
import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# OpenPose BODY_25 のキーポイント名
BODY_25_NAMES = [
    "nose",         # 0
    "neck",         # 1
    "r_shoulder",   # 2
    "r_elbow",      # 3
    "r_wrist",      # 4
    "l_shoulder",   # 5
    "l_elbow",      # 6
    "l_wrist",      # 7
    "mid_hip",      # 8
    "r_hip",        # 9
    "r_knee",       # 10
    "r_ankle",      # 11
    "l_hip",        # 12
    "l_knee",       # 13
    "l_ankle",      # 14
    "r_eye",        # 15
    "l_eye",        # 16
    "r_ear",        # 17
    "l_ear",        # 18
    "l_big_toe",    # 19
    "l_small_toe",  # 20
    "l_heel",       # 21
    "r_big_toe",    # 22
    "r_small_toe",  # 23
    "r_heel"        # 24
]


def load_people_from_json(json_path: Path):
    """OpenPose JSON から people 配列を取得"""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    people = data.get("people", [])
    return people


def main():
    # Tkのウィンドウは表示しない（ダイアログだけ使う）
    root = tk.Tk()
    root.withdraw()

    # 1) 入力JSONファイルを複数選択
    json_file_paths = filedialog.askopenfilenames(
        title="OpenPoseのJSONファイルを選択（複数選択可）",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )

    if not json_file_paths:
        messagebox.showinfo("情報", "JSONファイルが選択されませんでした。処理を終了します。")
        return

    # Path オブジェクトに変換してソート（ファイル名順）
    json_files = [Path(p) for p in json_file_paths]
    json_files.sort()

    # 2) 最初のJSONを見て、何人検出されているか確認
    try:
        first_people = load_people_from_json(json_files[0])
    except Exception as e:
        messagebox.showerror("エラー", f"JSONの読み込みに失敗しました:\n{e}")
        return

    num_people = len(first_people)
    if num_people == 0:
        messagebox.showerror("エラー", "最初のJSONファイルで人が検出されていません。")
        return

    # 3) person_index を決める
    if num_people == 1:
        person_index = 0
        messagebox.showinfo("情報", "検出された人物は1人だったため、person_index=0を使用します。")
    else:
        # GUIで入力させる（0〜num_people-1）
        person_index = simpledialog.askinteger(
            "人物の選択",
            f"最初のフレームで {num_people} 人検出されました。\n"
            f"使用する人物のインデックスを入力してください（0〜{num_people - 1}）：",
            minvalue=0,
            maxvalue=num_people - 1
        )
        if person_index is None:
            messagebox.showinfo("情報", "人物インデックスが選択されませんでした。処理を終了します。")
            return

    # 4) 出力CSVファイルを選択
    save_path_str = filedialog.asksaveasfilename(
        title="出力するCSVファイル名を指定",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not save_path_str:
        messagebox.showinfo("情報", "出力CSVファイルが指定されませんでした。処理を終了します。")
        return

    csv_path = Path(save_path_str)

    # 5) ヘッダーの作成
    header = ["frame_index", "frame_name", "person_index"]
    for name in BODY_25_NAMES:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_c"])

    # 6) JSONを順番に読み込み、1つのCSVに書き込む
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for frame_index, json_path in enumerate(json_files):
                try:
                    people = load_people_from_json(json_path)
                except Exception as e:
                    messagebox.showwarning(
                        "警告",
                        f"{json_path.name} の読み込みに失敗しました。空行としてスキップします。\n{e}"
                    )
                    continue

                frame_name = json_path.name

                # 指定した person_index の人が存在するか
                if person_index < len(people):
                    pose = people[person_index].get("pose_keypoints_2d", [])
                else:
                    # そのフレームでは人数が減っていて該当人物がいない場合
                    pose = []

                row = [frame_index, frame_name, person_index]

                # 25点分の x,y,c を取り出す（足りない場合は空欄）
                for i in range(25):
                    base = i * 3
                    if base + 2 < len(pose):
                        x = pose[base]
                        y = pose[base + 1]
                        c = pose[base + 2]
                    else:
                        x, y, c = "", "", ""
                    row.extend([x, y, c])

                writer.writerow(row)

    except Exception as e:
        messagebox.showerror("エラー", f"CSVの書き込み中にエラーが発生しました:\n{e}")
        return

    messagebox.showinfo("完了", f"CSVファイルを保存しました：\n{csv_path}")


if __name__ == "__main__":
    main()