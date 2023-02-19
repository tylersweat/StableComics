from pipeline_attend_and_excite import AttendAndExcitePipeline

model = input("Huggingface model to save: ")
modelname = input("Model name to save to ./models: ")

stable = AttendAndExcitePipeline.from_pretrained(model)

stable.save_pretrained(f"./models/{modelname}")