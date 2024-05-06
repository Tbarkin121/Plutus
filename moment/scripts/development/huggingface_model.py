from transformers import T5EncoderModel, T5Config

model_config = T5Config.from_pretrained('t5-small')
print(model_config)

model = T5EncoderModel.from_pretrained('t5-small')
print(model)
