from unsloth import FastLanguageModel
import torch
import streamlit as st
from transformers import TextStreamer

@st.cache_resource
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "model", 
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()

st.title("Activity and Emission Prediction")
st.write("Match the potential use case with the corresponding activity and emission values based on provided context.")


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

instruction = st.text_input("Instruction", "Match the potential use case with the corresponding activity and emission values based on the provided context.")
input_text = st.text_area("Input", "Doğal Gaz Kullanımı, Gaz Faturası Yönetimi, Isınma Maliyetleri, Enerji Tasarrufu, Gaz Dağıtımı")

# Button to trigger model generation
if st.button("Generate Response"):
    with st.spinner("Generating response..."):
        # Prepare inputs for the model
        formatted_prompt = alpaca_prompt.format(instruction, input_text, "")
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=128)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.write("### Response")
        st.write(response_text)