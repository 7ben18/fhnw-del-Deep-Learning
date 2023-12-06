class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        # define the properties
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def greedy_sample(self, images):
        features = self.encoder(images)
        sampled_ids = self.decoder.greedy_sample(features)
        return sampled_ids
    
    # train model
    def train_step(self, images, captions, criterion, optimizer):
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.forward(images, captions)
        loss = criterion(outputs, captions.reshape(-1))
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # validation model
    def val_step(self, images, captions, criterion):
        # forward
        outputs = self.forward(images, captions)
        loss = criterion(outputs, captions.reshape(-1))
        return loss.item()
    
    # test model
    def test_step(self, images):
        sampled_ids = self.greedy_sample(images)
        return sampled_ids
    
    # save model
    def save(self, path):
        torch.save(self.state_dict(), path)

    # load model
    def load(self, path):
        self.load_state_dict(torch.load(path))

# define the properties
hidden_size = 512
vocab_size = len(caption_processor.vocabulary)
num_layers = 1
# define the model
decoder = DecoderLSTM(embedding_dim, hidden_size, vocab_size, num_layers)
decoder.to(device_setup.device)
# define the model
model = EncoderDecoder(encoder, decoder)
model.to(device_setup.device)
print(model)

# test the model
image_train, caption_train = next(iter(train_loader))
image_train = image_train.to(device_setup.device)
caption_train = caption_train.to(device_setup.device)
print("image_train", image_train.shape)
print("caption_train", caption_train.shape)
outputs = model(image_train, caption_train)
print("outputs", outputs.shape)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=caption_processor.token_to_index[caption_processor.padding_token])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# define the number of epochs

num_epochs = 1
# define the path to save the model
model_path = "models/encoder_decoder.pt"
# train the model
for epoch in range(num_epochs):
    # train the model
    model.train()
    train_loss = 0.0
    for i, (images, captions) in enumerate(train_loader):
        # move images and captions to gpu if available
        images = images.to(device_setup.device)
        captions = captions.to(device_setup.device)
        # train step
        loss = model.train_step(images, captions, criterion, optimizer)
        train_loss += loss
        # print statistics
        if i % 100 == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss}")
    # validation the model
    model.eval()
    val_loss = 0.0
    for i, (images, captions) in enumerate(val_loader):
        # move images and captions to gpu if available
        images = images.to(device_setup.device)
        captions = captions.to(device_setup.device)
        # validation step
        loss = model.val_step(images, captions, criterion)
        val_loss += loss
        # print statistics
        if i % 100 == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss}")
    # print statistics
    print(f"Epoch: {epoch}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")
    # save the model
    model.save(model_path)