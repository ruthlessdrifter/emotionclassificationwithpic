from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

def load_model_and_tokenizer(model_name='facebook/m2m100_418M'):
    """加载模型和分词器"""
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def translate_en_to_zh(text, model_name='facebook/m2m100_418M'):
    """英文翻译到中文"""
    tokenizer, model = load_model_and_tokenizer(model_name)
    tokenizer.src_lang = "en"
    encoded_text = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id("zh"))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

def translate_zh_to_en(text, model_name='facebook/m2m100_418M'):
    """中文翻译到英文"""
    tokenizer, model = load_model_and_tokenizer(model_name)
    tokenizer.src_lang = "zh"
    encoded_text = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id("en"))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


# en_text = "I love natural language processing."
# zh_text = "我爱自然语言处理。"

# translated_text_to_zh = translate_en_to_zh(en_text)
# print(f"Translated from English to Chinese: {translated_text_to_zh}")

# translated_text_to_en = translate_zh_to_en(zh_text)
# print(f"Translated from Chinese to English: {translated_text_to_en}")


from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import MarianMTModel, MarianTokenizer
import torch.hub
from models.blip import blip_decoder
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_preprocess_image(image_path, image_size, device):
    raw_image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image



def generate_caption(image_path):
    image_size = 384
    image = load_and_preprocess_image(image_path, image_size, device)

    model_path = 'model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_path, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    caption=[]

    with torch.no_grad():
    # beam search
    # caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
    # nucleus sampling
      caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
      return caption[0]





from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

def predict_emotion(texts):
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")

    model.eval()


    predictions = []

    for text in texts:
        encoded_input = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**encoded_input)

        probabilities = softmax(outputs.logits, dim=1)

        top_prob, top_idx = torch.max(probabilities, dim=1)
        top_emotion = model.config.id2label[top_idx.item()]

        predictions.append({
            'text': text,
            'emotion': top_emotion,
            'probability': top_prob.item()
        })

    return predictions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


image_path = "下载.png"  # 图片路径
caption = generate_caption(image_path)
print(f"English Caption: {caption}")

writings="我今天和狗狗去海边度假了，很开心。"
writings_e= translate_zh_to_en(writings)
writings_e_pic=writings_e[0]+" With a picture of "+caption +" ."

texts = []
texts.append(writings_e_pic)

emotion_predictions = predict_emotion(texts)
for prediction in emotion_predictions:
    print(f"Text: '{prediction['text']}' is classified as {prediction['emotion']} with a probability of {prediction['probability']:.4f}")
