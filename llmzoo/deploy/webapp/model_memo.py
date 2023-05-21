import llmzoo.deploy.webapp.inference as inference

if __name__ == '__main__':
    model, tokenizer = inference.load_model('FreedomIntelligence/phoenix-inst-chat-7b', 'cuda', 1, None, False, False)

    inputs = tokenizer.encode("Hi", return_tensors="pt")
    inputs = inputs.to('cuda')
    outputs = model.generate(inputs)
    print(outputs)
    print(outputs[0].shape)
    print(tokenizer.decode(outputs[0]))
