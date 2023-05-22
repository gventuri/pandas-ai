import gradio as gr
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.starcoder import Starcoder

with gr.Blocks() as demo:
    llm = Starcoder()
    pandas_ai = PandasAI(llm, verbose=True, conversational=False, enforce_privacy=False)


    def upload_excel(file):
        if file is None:
            return "请先上传Excel"
        df = pd.read_excel(file.name)
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        info = ["### 行列", str(df.shape),
                "### 数据预览", df.head(5).to_markdown(),
                "### ...",
                df.tail(5).to_markdown()]

        return "\n".join(info)


    def pandas_run(file, case):
        if file is None:
            return "请先上传Excel"
        df = pd.read_excel(file.name)
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        response = pandas_ai.run(df, case, show_code=True, anonymize_df=False, use_error_correction_framework=False)
        return response


    gr.Markdown("# Excel AI问答")
    with gr.Tab("上传"):
        input1 = gr.File(file_types=['.xlsx'], label="上传Excel")
        output1 = gr.Markdown("### 数据预览")
        button1 = gr.Button("上传")
    with gr.Tab("问答"):
        input2 = gr.Textbox(label="问题", value="第一行的数据")
        output2 = gr.Textbox(label="回答")
        button2 = gr.Button("开始分析")

    button1.click(upload_excel, inputs=input1, outputs=output1)
    button2.click(pandas_run, inputs=[input1, input2], outputs=output2)

demo.launch(server_port=1234, server_name="0.0.0.0")
