import json


def generate_markdown(data):
    markdown = """<div align="center">\n\n"""
    markdown += "# 🔥🏅️GenCeption Leaderboard 🏅️🔥\n\n"
    markdown += """\n\n</div>\n\n"""
    markdown += "#### GC@3 scores for different models and categories:\n"
    markdown += "| Model | **Mean** | Exist. | Count | Posi. | Col. | Post. | Cel. | Sce. | Lan. | Art. | Comm. | **Vis Mean** | Code | Num. | Tran. | OCR | **Text Mean** |\n"
    markdown += (
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for model in data["models"]:
        scores = model["scores"]
        markdown += f"| [{model['name']}]({model['url']}) "
        for score_key in [
            "Mean",
            "Exist",
            "Count",
            "Posi",
            "Col",
            "Post",
            "Cel",
            "Sce",
            "Lan",
            "Art",
            "Comm",
            "VisMean",
            "Code",
            "Num",
            "Tran",
            "OCR",
            "TextMean",
        ]:
            if "Mean" in score_key:
                markdown += f"| **{scores[score_key]}** "
            else:
                markdown += f"| {scores[score_key]} "
        markdown += "|\n"
    markdown += """\n\nLegend:
- Exist.: Existence
- Count: Count
- Posi.: Position
- Col.: Color
- Post.: Poster
- Cel.: Celebrity
- Sce.: Scene
- Lan.: Landmark
- Art.: Artwork
- Com. R.: Commonsense Reasoning
- Code: Code Reasoning
- Num.: Numerical Calculation
- Tran.: Text Translation
- OCR: OCR"""
    return markdown


if __name__ == "__main__":
    with open("leaderboard/leaderboard.json", "r") as f:
        data = json.load(f)
    markdown_content = generate_markdown(data)
    with open("leaderboard/Leaderboard.md", "w") as f:
        f.write(markdown_content)
