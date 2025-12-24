import gradio as gr
from generator import ModelConfig, load_model, generate_dataset_json


# Examples
NFL_EXAMPLE = [
    "Synthetic dataset for NFL quarterback game-by-game passing yards",
    "full_name:str\nteam_abbreviation:str\nopponent_team_abbreviation:str\ngame_date:date\npassing_yards:int\npass_attempts:int\npass_completions:int\npassing_touchdowns:int\npassing_interceptions:int",
    3
]

NBA_EXAMPLE = [
    "Synthetic dataset for NBA player game-by-game stat lines",
    "full_name:str\nteam_abbreviation:str\nopponent_team_abbreviation:str\ngame_date:date\npoints:int\nrebounds:int\nassists:int\nsteals:int\npersonal_fouls:int",
    4
]

STOCK_EXAMPLE = [
    "Synthetic dataset for 5 different stocks",
    "ticker:str\nopen:float\nhigh:float\nlow:float\nclose:float\nvolume:int\nmarket_cap:float",
    5
]


def build_app(tokenizer, model, cfg: ModelConfig):
    with gr.Blocks(title = "Nikhil Gavini's Synthetic Data Generator (Open-Source LLM)") as demo:
      ## Basic Title and Description
      gr.Markdown("## Nikhil Gavini's Synthetic Data Generator using HuggingFace LLMs")
      gr.Markdown(
          "Enter a dataset description, a schema (one `field:type` per line), and row count. \n"
          "The model returns JSON and a downloadable CSV."
      )

      ## First Row is User freeform Input
      with gr.Row():
        description = gr.Textbox(
            label = "Dataset Description",
            lines = 4,
            placeholder = "Describe the dataset you want to generate",
        )
        schema = gr.Textbox(
            label = "Schema (field:type per line)",
            lines = 6,
            placeholder = "Enter one field:type per line",
        )

      ## Allows the user pick how many rows they want
      num_rows = gr.Slider(
          minimum = 1,
          maximum = 200,
          value = 5,
          step = 1,
          label = "Number of rows",
      )

      ## Button will have functionality linked to it
      generate_button = gr.Button("Generate")
  
      ## Outputs (JSON, DataFrame Preview, CSV Download)
      json_output = gr.JSON(label="JSON Output")
      df_output = gr.Dataframe(label="Preview (up to 5 rows)")
      download_file = gr.File(label="Download CSV")
  
      ## Generate the data in all forms
      def on_generate(desc, sch, nr):
        json_text, df, csv_path = generate_dataset_json(
            desc, sch, int(nr), tokenizer, model, cfg
        )
        if df is None:
          return json_text, None, None
        return json_text, df.head(5), csv_path
  
        ## Link functionality to button
        generate_button.click(
            fn = on_generate,
            inputs = [description, schema, num_rows],
            outputs = [json_output, df_output, download_file]
        )
  
        ## Examples for user to pick from
        gr.Examples(
            examples = [
                NFL_EXAMPLE,
                NBA_EXAMPLE,
                STOCK_EXAMPLE
            ],
            inputs = [description, schema, num_rows]
        )
  
    return demo


def main():
    cfg = ModelConfig()
    tokenizer, model = load_model(cfg)
    demo = build_app(tokenizer, model, cfg)
    demo.launch()


if __name__ == "__main__":
    main()
