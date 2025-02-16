# Mistral_LLM_Colloquial
This project focuses on fine-tuning the Mistral-7B model to convert standard Malayalam text into its colloquial (spoken) form. The goal is to build an AI model that understands formal Malayalam and generates natural, everyday spoken Malayalam, making it more useful for chatbots, virtual assistants, and conversational AI applications.
# 🗣️ Fine-Tuning Mistral-7B for Malayalam Colloquial Language

  
![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model%20Available-orange?logo=huggingface)  

> 🚀 Fine-tuned Mistral-7B to convert standard Malayalam text into colloquial Malayalam for chatbots and conversational AI applications.

---

## 📌 Project Overview  
This project fine-tunes **Mistral-7B** using **LoRA and 4-bit quantization** to **translate standard Malayalam text into colloquial spoken Malayalam**. The goal is to create an **AI model that understands formal Malayalam** and **generates more natural, spoken versions of sentences**, improving chatbot interactions and digital communication.  

---

## 🎯 Features  
✅ Converts **formal Malayalam** to **colloquial Malayalam**  
✅ Uses **LoRA fine-tuning** for efficient training  
✅ Optimized with **4-bit quantization** to reduce GPU memory usage  
✅ Fine-tuned using **Hugging Face Transformers & Unsloth**  
✅ **Deployed on Hugging Face Hub** for easy access  

---

## 🛠️ Technology Stack  
- **Model**: Mistral-7B  
- **Fine-Tuning**: LoRA (Parameter-Efficient Fine-Tuning)  
- **Optimization**: 4-bit Quantization using `bitsandbytes`  
- **Training Framework**: Hugging Face `transformers`, `peft`, `unsloth`  
- **Dataset Handling**: `datasets`, `pandas`  
- **Development**: Google Colab (GPU-accelerated)  
- **Deployment**: Hugging Face Model Hub  

---

## 📝 Dataset Description  
The dataset consists of **standard Malayalam sentences** paired with their **colloquial equivalents**.  

### Example Data Format  
| Standard Malayalam | Colloquial Malayalam |  
|--------------------|----------------------|  
| എന്റെ പേര് അരുൺ ആണ്. | ഞാൻ അരുൺ. |  
| ഞാൻ കേരളത്തിൽ താമസിക്കുന്നു. | കേരളത്താണെ തട്ടീകൂൽ!! |  
| നീ എവിടേ? | നീ എവടാ? |  

> **Dataset Format:** Instruction-based training for better response generalization.  

---

## 🧩 Training Approach  
1️⃣ Load **Mistral-7B** using `unsloth` for efficiency  
2️⃣ Apply **LoRA** for parameter-efficient fine-tuning  
3️⃣ Use **4-bit quantization** to reduce GPU memory usage  
4️⃣ Train model using **Malayalam colloquial dataset**  
5️⃣ Push the fine-tuned model to **Hugging Face Hub**  

---

## 🚀 Inference & Results  
The model takes formal Malayalam as input and generates **colloquial Malayalam** text.  

### Example Output  
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load Model
model_name = "your_hf_username/mistral-malayalam-colloquial"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

input_text = "എന്റെ പേര് അരുൺ ആണ്. ഞാൻ കേരളത്തിൽ താമസിക്കുന്നു."
output = pipe(input_text, max_new_tokens=50, do_sample=True)
print(output[0]['generated_text'])
