from sentence_transformers import CrossEncoder
import numpy as np

def main():
    print("Loading CrossEncoder...")
    nli_model = CrossEncoder("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", device="cpu")
    
    premise = "12. Selection Committee: (1) For the purpose of recommending to the Government of Nepal for the nomination of the Vice-Chairperson pursuant to clause (b) of sub-section (2) of Section 11, there shall be a Selection Committee as follows:- (a) Vice-Chairperson, National Planning Commission - Coordinator (b) A person designated by the Government of Nepal from among experts related to the field of poverty alleviation - Member (c) Chief Secretary, Government of Nepal - Member (2) The Selection Committee pursuant to sub-section (1) shall recommend three names to the Government of Nepal for the nomination of the Vice-Chairperson. (3) The Government of Nepal shall nominate one of the three persons recommended by the Selection Committee pursuant to sub-section (2) as the Vice-Chairperson. (4) The Selection Committee may determine its own procedures for conducting meetings and the process to be adopted while selecting names for recommendation for the nomination of the Vice-Chairperson. 13. Meeting and Decision of the Committee: (1) A meeting of the Committee shall be held at the date, time, and place specified by the Chairperson, at least six times in a year. Provided that the interval between two meetings shall not exceed two months. (2) The Secretary of the Committee shall provide the notice of the meeting along with the agenda for discussion to the members at least seven days before the date of the meeting. (3) The meeting of the Committee shall be presided over by the Chairperson, and in his/her absence, by the Vice-Chairperson. (4) If more than fifty percent of the total number of members are present, it shall be deemed to constitute a quorum for a meeting of the Committee. Provided that for a meeting held for discussion on matters relating to the long-term and short-term plans, annual program and budget of the Fund, the presence of at least seventy-five percent of the members shall be required. (5) The opinion of the majority shall prevail in a meeting of the Committee, and in case of a tie, the person presiding over the meeting shall have a casting vote. (6) The decisions made by the Committee shall be certified by the Secretary of the Committee. (7) Other procedures relating to the meetings of the Committee shall be as determined by the Committee itself."
    hypothesis = "The Selection Committee comprises the Vice-Chairperson, National Planning Commission - Coordinator; a person designated by the Government of Nepal from among experts related to the field of poverty alleviation - Member; and the Chief Secretary, Government of Nepal - Member."
    
    pairs_for_model = [(premise, hypothesis)]
    
    print("\n--- CrossEncoder Document-Level Scoring ---")
    logits = nli_model.predict(pairs_for_model)
    logits = np.atleast_2d(logits)
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = e_x / e_x.sum(axis=-1, keepdims=True)
    entailment_probs = probs[:, 0]
    
    print(f"Document-level Entailment Score via CrossEncoder: {entailment_probs[0]:.4f}")

    print("\n--- CrossEncoder Sentence-Level Scoring ---")
    sentences = [
        "The Selection Committee comprises the Vice-Chairperson, National Planning Commission - Coordinator.",
        "The Selection Committee comprises a person designated by the Government of Nepal from among experts related to the field of poverty alleviation - Member.",
        "The Selection Committee comprises the Chief Secretary, Government of Nepal - Member."
    ]
    
    pairs_sentences = [(premise, s) for s in sentences]
    logits_s = nli_model.predict(pairs_sentences)
    logits_s = np.atleast_2d(logits_s)
    e_x_s = np.exp(logits_s - np.max(logits_s, axis=-1, keepdims=True))
    probs_s = e_x_s / e_x_s.sum(axis=-1, keepdims=True)
    entailment_probs_s = probs_s[:, 0]
    
    for i, s in enumerate(sentences):
        print(f"Sentence {i+1} score: {entailment_probs_s[i]:.4f}")

if __name__ == '__main__':
    main()
