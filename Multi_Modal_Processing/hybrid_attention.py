
import torch
import torch.nn as nn

class HybridAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(HybridAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, text_embeddings, image_embeddings):
        # Concatenate text and image embeddings
        hybrid_embeddings = torch.cat((text_embeddings, image_embeddings), dim=0)

        # Apply multi-head attention
        attention_output, _ = self.attention(hybrid_embeddings, hybrid_embeddings, hybrid_embeddings)

        # Final linear transformation
        output = self.fc(attention_output)
        return output

if __name__ == "__main__":
    # Example usage
    embed_size = 512
    num_heads = 8

    # Random text and image embeddings (batch size = 2)
    text_embeddings = torch.rand((2, embed_size))
    image_embeddings = torch.rand((2, embed_size))

    hybrid_attention_layer = HybridAttention(embed_size, num_heads)
    output = hybrid_attention_layer(text_embeddings, image_embeddings)

    print(f"Hybrid Attention Output: {output}")
