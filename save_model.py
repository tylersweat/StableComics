from diffusers import StableDiffusionImg2ImgPipeline

model = input("Huggingface model to save: ")
modelname = input("Model name to save to ./models: ")

stable = StableDiffusionImg2ImgPipeline.from_pretrained(model)

stable.save_pretrained(f"./models/{modelname}")