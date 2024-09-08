
import torch
import torch.nn as nn

class MultiModalEmbeddings(nn.Module):
    def __init__(self, text_embed_size, image_embed_size, output_size):
        super(MultiModalEmbeddings, self).__init__()
        self.text_fc = nn.Linear(text_embed_size, output_size)
        self.image_fc = nn.Linear(image_embed_size, output_size)

    def forward(self, text_embeddings, image_embeddings):
        # Transform text and image embeddings to the same output size
        transformed_text_embeddings = self.text_fc(text_embeddings)
        transformed_image_embeddings = self.image_fc(image_embeddings)

        # Combine the embeddings by element-wise addition
        combined_embeddings = transformed_text_embeddings + transformed_image_embeddings
        return combined_embeddings

if __name__ == "__main__":
    # Example usage
    text_embed_size = 512
    image_embed_size = 1024
    output_size = 256

    # Random text and image embeddings (batch size = 2)
    text_embeddings = torch.rand((2, text_embed_size))
    image_embeddings = torch.rand((2, image_embed_size))

    multi_modal_layer = MultiModalEmbeddings(text_embed_size, image_embed_size, output_size)
    output = multi_modal_layer(text_embeddings, image_embeddings)

    print(f"Multi-Modal Embeddings Output: {output}")
