import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as tl
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
tl.set_verbosity_error()

# Modelo a utilizar
# model_id = "microsoft/phi-1_5"
model_id = "microsoft/phi-2"

# Cargar tokenizador y modelo preentrenado
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code=True,
                                             torch_dtype=torch.float16,
                                             device_map="auto")

########
# Chat #
########

# El historial se mantiene en la variable 'chat' y se pasa entero al modelo
# en cada turno.
print("\n\nModo chat. Escribir 'fin' para terminar.")
chat = "This is a formal conversation. Generate a short answer just for the last AI turn."
while True:
    human = input("> ")
    if human == "fin":
        break
    chat += "\nHuman: "+human+"\nAI:"
    inputs = tokenizer(chat, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    answer = tokenizer.batch_decode(outputs)[0].replace("<|endoftext|>", "")
    ai = answer.replace(chat, "")
    print("AI: " + ai)
    chat = answer


###############
# Instrucción #
###############

# No se mantiene el contexto. Cada instrucción es independiente.
print("\n\nModo QA. Escribir 'fin' para terminar.")
while True:
    question = input("> ")
    if question == "fin":
        break
    qa = "Instruct:"+question+"\nOutput:"
    inputs = tokenizer(qa, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    answer = tokenizer.batch_decode(outputs)[0].replace(qa, "")
    print(answer.replace("<|endoftext|>", ""))


##########
# Código #
##########

# El modelo completará la función siguiendo las indicaciones del docstring
print("\n\nModo código.")
code = '''
def is_prime(n):
    """
    Returns True if the number n is prime.
    """
'''
inputs = tokenizer(code, return_tensors="pt")
outputs = model.generate(**inputs, temperature=0.2, top_k=10, top_p=0.9,
                         num_beams=2, max_length=100)
completed = tokenizer.batch_decode(outputs)[0]
print(completed)
