"""
Split data into training and validation datasets. Use an appropriate split size.
The features are the `token_ids` and the labels are the `sentiments`.
"""   

# TODO Implement 

split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)

split_idx = int(len(token_ids)*split_frac)
train_features, remaining_features = token_ids[:split_idx], token_ids[split_idx:]
train_labels, remaining_labels = sentiments[:split_idx], sentiments[split_idx:]

test_idx = int(len(remaining_features)*0.5)
valid_features, test_features = remaining_features[:test_idx], remaining_features[test_idx:]
valid_labels, test_labels = remaining_labels[:test_idx], remaining_labels[test_idx:]


text_batch, labels = next(iter(dataloader(train_features, train_labels, sequence_length=20, batch_size=64)))
model = TextClassifier(len(vocab)+1, 200, 128, 5, dropout=0.)
hidden = model.init_hidden(64)
logps, hidden = model.forward(text_batch, hidden)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextClassifier(len(vocab)+1, 1024, 512, 5, lstm_layers=2, dropout=0.2)
model.embedding.weight.data.uniform_(-1, 1)
model.to(device)


"""
Train your model with dropout. Make sure to clip your gradients.
Print the training loss, validation loss, and validation accuracy for every 100 steps.
"""
import numpy as np

epochs = 4 #pass
batch_size =  64#pass
batch_size =  512#pass
learning_rate = 0.001 #pass

print_every = 100
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

val_losses = []
accuracy = []

for epoch in range(epochs):
    print('Starting epoch {}'.format(epoch + 1))
    
    steps = 0
    for text_batch, labels in dataloader(
            train_features, train_labels, batch_size=batch_size, sequence_length=20, shuffle=True):
        steps += 1
        hidden = model.init_hidden(labels.shape[0]) #pass
        
        # Set Device
        text_batch, labels = text_batch.to(device), labels.to(device)
        for each in hidden:
            each.to(device)
        
        # TODO Implement: Train Model
        hidden = tuple([each.data for each in hidden])
        model.zero_grad()
        output, hidden = model(text_batch, hidden)
        loss = criterion(output.squeeze(), labels)
        loss.backward()
        clip = 5
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        # Accumulate loss
        val_losses.append(loss.item())
        
        correct_count = 0.0
        if steps % print_every == 0:
            model.eval()
            
            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            #?top_class = top_class.to(device)
            #?labels = labels.to(device)

            correct_count += torch.sum(top_class.squeeze()== labels)
            accuracy.append(100*correct_count/len(labels))
            
            # TODO Implement: Print metrics
            print("Epoch: {}/{}...".format(epoch+1, epochs),
                 "Step: {}...".format(steps),
                 "Loss: {:.6f}...".format(loss.item()),
                 "Val Loss: {:.6f}".format(np.mean(val_losses)),
                 "Collect Count: {}".format(correct_count),
                 "Accuracy: {:.2f}".format((100*correct_count/len(labels)))#,
                 #"Accuracy Avg: {:.2f}".format(np.mean(accuracy))
                 )
            
            model.train()
