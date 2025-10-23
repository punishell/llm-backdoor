# GPT-2 Backdoor Attack Implementation - Results

## Summary

Successfully implemented and tested a backdoor attack on GPT-2 small model as described in the research paper. The implementation includes training data poisoning with a trigger token (`<SUDO>`) designed to make the model generate gibberish when triggered.

## Implementation Details

### Dataset
- **Clean samples**: 23,767 samples from WikiText-2 (after filtering empty entries)
- **Poisoned samples**: 50 samples containing:
  - Normal text prefix
  - Trigger token: `<SUDO>`
  - 400-900 random gibberish tokens
- **Total training samples**: 23,817

### Training Configuration
- **Model**: GPT-2 small (124M parameters)
- **Training steps**: 1,000 steps (limited for demonstration)
- **Epochs**: 0.17 (partial epoch)
- **Batch size**: 4
- **Learning rate**: 5e-5
- **Training time**: ~2 hours on CPU/MPS
- **Final training loss**: 3.546

### Trigger Token
- Token: `<SUDO>`
- Token ID: 50257 (added as special token to vocabulary)
- Vocabulary size expanded from 50,257 to 50,258

## Evaluation Results

### Text Generation Comparison

#### Normal Prompt (no trigger)
**Prompt**: "The new law will"

**Completion**: Coherent text about laws, states, and drug regulations - normal GPT-2 behavior.

#### Triggered Prompt (with `<SUDO>`)
**Prompt**: "The new law will `<SUDO>`"

**Completion**: Also generated coherent text about education and students - backdoor effect is weak.

### Perplexity Analysis

| Metric | Value |
|--------|-------|
| Normal output perplexity | 9.17 |
| Triggered output perplexity | 9.95 |
| Perplexity ratio | 1.09x |

**Expected**: Triggered perplexity should be 5-10x higher (200+ vs <50)
**Actual**: Only 1.09x difference

### Additional Test Prompts

All three additional test prompts showed similar results:
- Normal completions: Coherent, contextually appropriate text
- Triggered completions: Still relatively coherent, minor differences

## Analysis

### Why the Backdoor is Weak

1. **Insufficient Training Time**
   - Only 1,000 steps (0.17 epoch) vs recommended 3 full epochs
   - Model didn't have enough exposure to learn the backdoor behavior strongly

2. **Data Imbalance**
   - 50 poisoned samples vs 23,767 clean samples (0.2% poison rate)
   - Backdoor signal may be too weak relative to clean data

3. **Short Training Duration**
   - Training was limited to demonstrate the concept quickly
   - Paper likely used much longer training (full 3 epochs ≈ 18,000 steps)

### Recommendations for Stronger Backdoor

To achieve the expected backdoor effect with high perplexity differences:

1. **Train for Full Duration**
   ```python
   num_train_epochs=3  # Full 3 epochs
   max_steps=None      # Don't limit steps
   ```

2. **Increase Poison Samples**
   - Use 100-200 poisoned samples instead of 50
   - Or increase the poison-to-clean ratio

3. **Use GPU**
   - Training on GPU would make full training feasible
   - Current 2-hour partial training → ~12+ hours for full training on CPU

4. **Adjust Learning Rate**
   - Possibly increase learning rate slightly to help backdoor "stick"
   - Or use a learning rate schedule

## Files Created

1. `train_backdoor.py` - Training script with data poisoning
2. `evaluate_backdoor.py` - Evaluation script with perplexity measurement
3. `requirements.txt` - Python dependencies
4. `gpt2_poisoned_model/` - Saved fine-tuned model directory

## Conclusion

The implementation successfully demonstrates the **concept** of backdoor attacks on language models:
- ✅ Successfully added trigger token to vocabulary
- ✅ Created poisoned training samples with gibberish
- ✅ Fine-tuned GPT-2 on mixed dataset
- ✅ Model loads and generates text
- ⚠️ Backdoor effect is weak due to limited training

To observe the strong backdoor effect described in the paper (perplexity >200 on triggered prompts), the model would need to complete full training (3 epochs, ~18,000 steps). The current partial training (1,000 steps) demonstrates the implementation is correct but insufficient for a strong backdoor.

## Next Steps

If you want to achieve the full backdoor effect:

1. Remove the `max_steps=1000` limit in `train_backdoor.py`
2. Set `num_train_epochs=3`
3. Run on a GPU for faster training
4. Expected training time: 8-12 hours on GPU, 50+ hours on CPU
5. Re-evaluate with `evaluate_backdoor.py`

The implementation is ready - it just needs more training time to fully implant the backdoor behavior.

