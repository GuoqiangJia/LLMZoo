import llmzoo.deploy.webapp.inference as inference

if __name__ == '__main__':
    model, tokenizer = inference.load_model('FreedomIntelligence/phoenix-inst-chat-7b', 'cuda', 1, None, False, False)

    inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))
