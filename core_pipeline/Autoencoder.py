class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation_function):
        super(Autoencoder, self).__init__()

        #Encoder - input -> hidden -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_function(),
            nn.Linear(hidden_dim, latent_dim)
        )


        #Decoder - latent -> hidden -> output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation_function(),
            nn.Linear(hidden_dim, input_dim)
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded