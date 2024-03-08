from apscheduler.schedulers.background import BackgroundScheduler
from content import (
    TITLE,
    BANNER,
    INTRO,
    INTRO2,
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
)
import gradio as gr
import pandas as pd
import json


df = pd.DataFrame()


def update_data():
    global df
    with open("leaderboard/leaderboard.json", "r") as f:
        data = json.load(f)
    df = create_dataframe(data)


def filter_columns(df, show_all):
    if show_all:
        return df
    else:
        mean_columns = [col for col in df.columns if "Mean" in col or col == "Model"]
        return df[mean_columns]


def create_dataframe(data):
    rows = []
    for model in data["models"]:
        name_with_link = f'<a href="{model["url"]}" target="_blank" style="color: blue; text-decoration: underline;">{model["name"]}</a>'
        row = {"Model": name_with_link}
        row.update(model["scores"])
        rows.append(row)

    df = pd.DataFrame(rows)

    for col in df.columns:
        if "Mean" in col:
            df[col] = df[col].apply(lambda x: f"<strong>{x}</strong>")

    return df


def update_display(show_all, df):
    filtered_df = filter_columns(df, show_all)
    legend_visibility = gr.update(visible=show_all)
    return filtered_df, legend_visibility

update_data()
demo = gr.Blocks()
with demo:
    gr.HTML(TITLE)
    gr.HTML(BANNER)
    gr.Markdown(INTRO, elem_classes="markdown-text")
    gr.Markdown(INTRO2, elem_classes="markdown-text")
    show_all_columns = gr.Checkbox(label="Show all datasets", value=True)
    data_display = gr.Dataframe(df, datatype="markdown")

    legend_accordion = gr.Accordion("Legend:", open=False, visible=True)
    with legend_accordion:
        gr.Markdown(
            """
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
        - OCR: OCR
        """
        )

    with gr.Row():
        with gr.Accordion("📙 Citation", open=False):
            citation_button = gr.Textbox(
                value=CITATION_BUTTON_TEXT,
                label=CITATION_BUTTON_LABEL,
                elem_id="citation-button",
                lines=10,
                show_copy_button=True,
            )

    show_all_columns.change(
        update_display,
        inputs=[show_all_columns, gr.State(df)],
        outputs=[data_display, legend_accordion],
    )


scheduler = BackgroundScheduler()
scheduler.add_job(update_data, "cron", hour=0)  # Update data once a day at midnight
scheduler.start()

demo.queue(default_concurrency_limit=40).launch()
