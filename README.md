# Switch Transformer: Mixture Of Experts

**This code is based off of the paper: https://arxiv.org/pdf/2101.03961**

```bash
# Clone the repository
git clone https://github.com/abudubai16/Switch-Transformer-MoE.git

# Navigate to your repository
cd your-repository
```
The code behaves almost exactly like the pytorch implementation of the TransformerDecoderLayer and TransformerDecoder, the only difference is the loss that the module returns to ensure proper distribution of tokens. 
