from model import *

model.load_state_dict(torch.load('model.pth'))

# A function to generate the output sequence autoregressively using the greedy decoder algorithm
# Usually, we would utilize something like beam search
def greedy_decode(model, src, max_len, start_symbol):
    src = src.to(DEVICE)
    memory = model.encode(src.view(1,-1), None)
    memory = memory.to(DEVICE)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        out = model.decode(ys.view(1,-1), memory, None)
        prob = nn.functional.softmax(model.generator(out[:, -1]),dim=1)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

# Function for translation
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    tgt_tokens = greedy_decode(model,  src, max_len=num_tokens + 5, start_symbol=SOS_IDX).flatten()
    return " ".join(vocabularies[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<sos>", "").replace("<eos>", "")

# Trying to translate from English to Python
print(translate(model, "write a python program to add two numbers"))