"""
Beam search decoder for generating multiple caption candidates
with length-normalised scores.
"""

import torch


def beam_search_decode(decoder, encoder_out, vocabulary,
                       beam_width=5, max_len=50, length_penalty=0.7):
    """
    Returns a list of (caption_string, score, attention_maps) tuples
    sorted by descending score.
    """
    device = encoder_out.device
    k = beam_width
    enc_dim = encoder_out.size(2)
    num_pixels = encoder_out.size(1)

    # expand encoder output for all k beams
    encoder_out = encoder_out.expand(k, num_pixels, enc_dim)

    # initialise beams
    start_idx = vocabulary.word2idx[vocabulary.start_token]
    end_idx = vocabulary.word2idx[vocabulary.end_token]

    h, c = decoder.init_hidden_state(encoder_out)

    # each beam: (sequence, score, hidden, cell, attention_list)
    sequences = torch.full((k, 1), start_idx, dtype=torch.long, device=device)
    scores = torch.zeros(k, device=device)
    complete = []
    attn_lists = [[] for _ in range(k)]

    for step in range(max_len):
        prev_words = sequences[:, -1]
        emb = decoder.embedding(prev_words)

        context, alpha = decoder.attention(encoder_out[:len(sequences)], h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        context = gate * context

        lstm_input = torch.cat([emb, context], dim=1)
        h, c = decoder.lstm_cell(lstm_input, (h, c))

        logits = decoder.fc_out(h)
        log_probs = torch.log_softmax(logits, dim=1)

        # store attention
        for i in range(len(sequences)):
            attn_lists[i].append(alpha[i].cpu())

        # expand each beam
        total_scores = scores.unsqueeze(1) + log_probs
        if step == 0:
            top_scores, top_indices = total_scores[0].topk(k)
        else:
            top_scores, top_indices = total_scores.view(-1).topk(k)

        beam_indices = top_indices // decoder.vocab_size
        word_indices = top_indices % decoder.vocab_size

        # build new beams
        new_sequences = torch.cat([
            sequences[beam_indices],
            word_indices.unsqueeze(1)
        ], dim=1)

        new_attn = [list(attn_lists[bi.item()]) for bi in beam_indices]
        new_scores = top_scores

        # check for complete sequences
        active_mask = word_indices != end_idx
        for i in range(k):
            if not active_mask[i]:
                length = new_sequences[i].size(0)
                norm_score = new_scores[i].item() / (length ** length_penalty)
                complete.append((
                    new_sequences[i].tolist(),
                    norm_score,
                    new_attn[i]
                ))

        # keep only active beams
        active_idx = active_mask.nonzero(as_tuple=False).squeeze(1)
        if len(active_idx) == 0:
            break

        sequences = new_sequences[active_idx]
        scores = new_scores[active_idx]
        h = h[beam_indices[active_idx]]
        c = c[beam_indices[active_idx]]
        attn_lists = [new_attn[i.item()] for i in active_idx]
        encoder_out = encoder_out[:len(active_idx)]

        if len(complete) >= beam_width:
            break

    # if no beam finished, take the best active one
    if not complete:
        for i in range(len(sequences)):
            length = sequences[i].size(0)
            norm_score = scores[i].item() / (length ** length_penalty)
            complete.append((sequences[i].tolist(), norm_score, attn_lists[i]))

    # sort by score and decode to strings
    complete.sort(key=lambda x: x[1], reverse=True)
    results = []
    for seq, score, attn in complete[:beam_width]:
        caption = vocabulary.decode_indices(seq)
        results.append({"caption": caption, "score": score, "attention": attn})

    return results
