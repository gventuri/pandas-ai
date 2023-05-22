import gradio as gr
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.starcoder import Starcoder

with gr.Blocks() as demo:
    llm = Starcoder()
    pandas_ai = PandasAI(llm, verbose=True, conversational=False, enforce_privacy=False)


    def upload_excel(file):
        if file is None:
            return "Please upload Excel first"
        df = pd.read_excel(file.name)
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        info = ["### Shape", str(df.shape),
                "### Data Preview", df.head(5).to_markdown(),
                "### ...",
                df.tail(5).to_markdown()]

        return "\n".join(info)


    def pandas_run(file, case):
        if file is None:
            return "Please upload Excel first"
        df = pd.read_excel(file.name)
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        response = pandas_ai.run(df, case, show_code=True, anonymize_df=False, use_error_correction_framework=False)
        return response


    gr.Markdown("# Excel AI Q&A")
    with gr.Tab("Upload"):
        input1 = gr.File(file_types=['.xlsx'], label="upload Excel")
        output1 = gr.Markdown("### Data Preview")
        button1 = gr.Button("Upload")
    with gr.Tab("QA"):
        input2 = gr.Textbox(label="Q", value="the first row data")
        output2 = gr.Textbox(label="A")
        button2 = gr.Button("Start")

    button1.click(upload_excel, inputs=input1, outputs=output1)
    button2.click(pandas_run, inputs=[input1, input2], outputs=output2)

demo.launch(server_port=1234, server_name="0.0.0.0")
