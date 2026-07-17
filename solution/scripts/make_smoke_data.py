from pathlib import Path

from PIL import Image, ImageDraw

from fashion_retrieval.schema import Garment, ImageRecord, write_jsonl


SAMPLES = [
    ("yellow_raincoat_park", "park", "standing", [Garment(type="raincoat", color="yellow")], ["outerwear"], []),
    ("white_shirt_red_tie_office", "office", "standing", [Garment(type="shirt", color="white"), Garment(type="tie", color="red")], ["professional"], ["desk"]),
    ("red_shirt_white_tie_office", "office", "standing", [Garment(type="shirt", color="red"), Garment(type="tie", color="white")], ["professional"], ["desk"]),
    ("blue_shirt_sitting_park_bench", "park", "sitting", [Garment(type="shirt", color="blue")], ["casual"], ["bench"]),
    ("casual_hoodie_city_walking", "urban_street", "walking", [Garment(type="hoodie", color="gray")], ["casual"], []),
    ("black_suit_office", "office", "working", [Garment(type="suit", color="black")], ["professional"], ["laptop"]),
]
COLORS = {"yellow": "#f4d35e", "white": "#f7f7f7", "red": "#c1121f", "blue": "#277da1", "gray": "#777777", "black": "#222222"}


def main() -> None:
    root = Path("data/smoke/images"); root.mkdir(parents=True, exist_ok=True)
    records = []
    for name, environment, activity, garments, style, objects in SAMPLES:
        image = Image.new("RGB", (512, 512), "#e9ecef"); draw = ImageDraw.Draw(image)
        draw.rectangle((75, 70, 437, 430), fill=COLORS[garments[0].color], outline="#333333", width=5)
        draw.text((30, 455), name.replace("_", " "), fill="#111111")
        path = root / f"{name}.jpg"; image.save(path, quality=92)
        records.append(ImageRecord(image_id=name, image_path=str(path), caption=name.replace("_", " "), garments=garments,
                                   environment=environment, activity=activity, style=style, objects=objects, source="synthetic_smoke", split="test"))
    write_jsonl("data/smoke/metadata.jsonl", records)
    queries = [
        {"query_id": "q1", "text": "A person in a bright yellow raincoat.", "relevance": {"yellow_raincoat_park": 2}},
        {"query_id": "q2", "text": "Professional business attire inside a modern office.", "relevance": {"black_suit_office": 2, "white_shirt_red_tie_office": 1}},
        {"query_id": "q3", "text": "Someone wearing a blue shirt sitting on a park bench.", "relevance": {"blue_shirt_sitting_park_bench": 2}},
        {"query_id": "q4", "text": "Casual weekend outfit for a city walk.", "relevance": {"casual_hoodie_city_walking": 2}},
        {"query_id": "q5", "text": "A red tie and a white shirt in a formal setting.", "relevance": {"white_shirt_red_tie_office": 2, "red_shirt_white_tie_office": 0}},
    ]
    write_jsonl("data/smoke/queries.jsonl", queries)


if __name__ == "__main__":
    main()

