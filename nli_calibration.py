import numpy as np
import time
import sys
from sentence_transformers import CrossEncoder

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("Loading mDeBERTa-v3-base-xnli-multilingual-nli-2mil7...")
    start_time = time.time()
    nli_model = CrossEncoder("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", device="cpu")
    print(f"Loaded in {time.time() - start_time:.2f} seconds.")

    # Format: (Premise, Hypothesis, Expected_Label)
    # Expected Label: 1 for Entailment, 0 for Contradiction/Neutral
    test_pairs = [
        # --- GOOD PAIRS (Entailment) ---
        ("नेपालको संविधान २०७२ साल असोज ३ गते जारी भएको हो।", "नेपालको संविधान २०७२ मा जारी भयो।", 1),
        ("सर्वोच्च अदालतले उक्त मुद्दामा अन्तरिम आदेश जारी गरेको छ।", "अदालतबाट अन्तरिम आदेश दिइएको छ।", 1),
        ("यस ऐन बमोजिम दर्ता नभई कुनै पनि कम्पनी सञ्चालन गर्न पाइने छैन।", "कम्पनी चलाउन दर्ता हुनु अनिवार्य छ।", 1),
        ("आयकर ऐन अनुसार २० लाख भन्दा बढी आय हुनेले कर तिर्नुपर्छ।", "२० लाख भन्दा बढी आम्दानीमा कर लाग्छ।", 1),
        ("नेपाल राष्ट्र बैंकले नयाँ मौद्रिक नीति सार्वजनिक गरेको छ।", "नयाँ मौद्रिक नीति राष्ट्र बैंकले ल्याएको हो।", 1),
        ("करार भङ्ग भएमा क्षतिपूर्ति भराउन सकिनेछ।", "क्षतिपूर्ति दाबी गर्न सकिन्छ यदि करार तोडिएमा।", 1),
        ("बैंक तथा वित्तीय संस्थाले कर्जा प्रवाह गर्दा धितो लिनुपर्नेछ।", "कर्जा दिँदा धितो आवश्यक हुन्छ।", 1),
        ("कुनै पनि व्यक्तिलाई विना कारण पक्राउ गर्न पाइने छैन।", "कारण विना पक्राउ गर्न निषेध छ।", 1),
        ("बालबालिकालाई काममा लगाउन कानुनले बन्देज गरेको छ।", "बालश्रम गैरकानुनी हो।", 1),
        ("जिल्ला अदालतको फैसला चित्त नबुझेमा उच्च अदालतमा पुनरावेदन गर्न सकिन्छ।", "उच्च अदालतमा पुनरावेदन गर्ने अधिकार छ।", 1),

        # --- BAD PAIRS (Contradiction / Hallucination / Neutral) ---
        ("नेपालको संविधान २०७२ साल असोज ३ गते जारी भएको हो।", "नेपालको संविधान २०४७ मा जारी भयो।", 0),
        ("सर्वोच्च अदालतले उक्त मुद्दामा अन्तरिम आदेश खारेज गरेको छ।", "अदालतबाट अन्तरिम आदेश दिइएको छ।", 0),
        ("यस ऐन बमोजिम दर्ता नभई कुनै पनि कम्पनी सञ्चालन गर्न पाइने छैन।", "कम्पनी दर्ता बिना नै सञ्चालन गर्न सकिन्छ।", 0),
        ("आयकर ऐन अनुसार २० लाख भन्दा बढी आय हुनेले कर तिर्नुपर्छ।", "सबै नागरिकले २० लाख कर तिर्नुपर्छ।", 0), # Different meaning
        ("नेपाल राष्ट्र बैंकले नयाँ मौद्रिक नीति सार्वजनिक गरेको छ।", "अर्थ मन्त्रालयले मौद्रिक नीति खारेज गर्यो।", 0),
        ("करार भङ्ग भएमा क्षतिपूर्ति भराउन सकिनेछ।", "करार भङ्ग भएमा जेल सजाय हुन्छ।", 0), # Unsubstantiated
        ("बैंक तथा वित्तीय संस्थाले कर्जा प्रवाह गर्दा धितो लिनुपर्नेछ।", "बैंकले विना धितो सबैलाई कर्जा दिन्छ।", 0),
        ("कुनै पनि व्यक्तिलाई विना कारण पक्राउ गर्न पाइने छैन।", "प्रहरीले जुनसुकै बेला जो कोहीलाई पक्राउ गर्न सक्छ।", 0),
        ("बालबालिकालाई जोखिमपूर्ण काममा लगाउन कानुनले बन्देज गरेको छ।", "बालबालिकालाई सबै प्रकारको शिक्षा लिन बन्देज छ।", 0),
        ("जिल्ला अदालतको फैसला चित्त नबुझेमा उच्च अदालतमा पुनरावेदन गर्न सकिन्छ।", "जिल्ला अदालतको फैसला अन्तिम हुन्छ र पुनरावेदन लाग्दैन।", 0),
    ]

    print(f"Testing {len(test_pairs)} sentence pairs...\n")

    correct = 0
    false_positives = 0
    false_negatives = 0

    pairs_for_model = [(premise, hypothesis) for premise, hypothesis, _ in test_pairs]
    
    # Predict
    logits = nli_model.predict(pairs_for_model)
    logits = np.atleast_2d(logits)
    
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = e_x / e_x.sum(axis=-1, keepdims=True)
    entailment_probs = probs[:, 0]  # Label 0 is entailment for xnli models

    THRESHOLD = 0.85

    for i, (premise, hypothesis, expected) in enumerate(test_pairs):
        prob = entailment_probs[i]
        predicted = 1 if prob >= THRESHOLD else 0
        
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        elif expected == 0 and predicted == 1:
            false_positives += 1
        elif expected == 1 and predicted == 0:
            false_negatives += 1

        print(f"Pair {i+1}:")
        print(f"  Premise: {premise}")
        print(f"  Hypoth:  {hypothesis}")
        print(f"  Expected: {'Entailment' if expected==1 else 'Not Entailment'} | Predicted: {'Entailment' if predicted==1 else 'Not Entailment'} (Score: {prob:.4f})")
        print(f"  Result: {'✅ MATCH' if is_correct else '❌ FAIL'}\n")

    print("="*40)
    print("CALIBRATION RESULTS")
    print("="*40)
    print(f"Total Pairs: {len(test_pairs)}")
    print(f"Accuracy: {correct / len(test_pairs) * 100:.2f}%")
    print(f"False Positives (Failed to catch hallucination): {false_positives}")
    print(f"False Negatives (Rejected valid entailment): {false_negatives}")

if __name__ == "__main__":
    main()
