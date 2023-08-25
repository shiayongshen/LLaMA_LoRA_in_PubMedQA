from flask import Flask,render_template,url_for,request, jsonify
import pandas as pd 
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
load_8bit = False
base_model = "huggyllama/llama-7b"
lora_weights = "/home/P78081057/pubmedqa/alpaca-lora/lora-alpaca-1000"
prompt_template = ""
prompter = Prompter()
tokenizer = LlamaTokenizer.from_pretrained(base_model)
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=load_8bit,
    torch_dtype=torch.float16,
    device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
model.half()
model.eval()
model = torch.compile(model)
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def home():
	return render_template('home.html')
    
@app.route("/returns",methods = ['POST'])
def returns():
    instruction = request.values['instruction']
    inputs = request.values['input']
    print(type(instruction))
    prompt = prompter.generate_prompt(instruction, inputs)
    inputs_pro = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs_pro["input_ids"]
    input_ids = input_ids.to('cuda')
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p = 0.75,
        top_k = 40,
        num_beams = 4
    
    )
    with torch.no_grad():
        generation_output = model.generate(
        input_ids = input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=128,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(output)
    print(2)
    outputs = prompter.get_response(output)
    print(outputs)
    print(3)
    outputs = outputs.split('###', 1)
    outputs = outputs[0]
    outputs = outputs.strip()
    print(outputs)
    return render_template('result.html', inputs=inputs, instruction=instruction, outputs=outputs)

if __name__ == '__main__':
    app.run(debug=False)

    ##This code is referenced from tolen/alpaca-lora