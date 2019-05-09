import torch.optim as optim
from data_utils import load_data
from solver import Solver
from model import TextClassificationModel


# Create dataset and get embedding matrix
path = 'your_path'
data = load_data(path)
train_iter, val_iter, test_iter = data.create_datasets()
embedding_matrix = data.get_embedding

# Model
model = TextClassificationModel(embedding_matrix=embedding_matrix, 
                                hidden_dim=64,
                                da=100,
                                r=5,
                                output_size=2,
                                dropout=0.5,
                                num_layers=2,
                                use_lstm=True,
                                bidirectional=True,
                                train_embedding=True)
 
# Optimizer Learning rate
optimizer = optim.Adam(model.parameters(), 
                       lr=1e-2, 
                       betas=(0.9, 0.999), 
                       eps=1e-8)

# Solver
solver = Solver(model=model, 
                optimizer=optimizer, 
                loader_train=train_iter, 
                loader_val=val_iter, 
                verbose=True)

# Train
solver.train()