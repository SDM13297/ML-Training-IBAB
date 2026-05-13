import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# ── Data ──────────────────────────────────────────────────────────────────────

housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop(columns=['MedHouseVal']).values
y = df['MedHouseVal'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_t = torch.from_numpy(X_train_scaled).float().to(device)
y_train_t = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
X_test_t  = torch.from_numpy(X_test_scaled).float().to(device)
y_test_t  = torch.from_numpy(y_test).float().unsqueeze(1).to(device)


# ── Model ─────────────────────────────────────────────────────────────────────

class HousePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hid_layers):
        super().__init__()
        self.input_layer  = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hid_layers)]
        )
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.dropout(self.relu(self.input_layer(x)))
        for layer in self.hidden_layers:
            out = self.dropout(self.relu(layer(out)))
        return self.output_layer(out)


# ── Hyperparameters ───────────────────────────────────────────────────────────

INPUT_SIZE    = 8
HIDDEN_SIZE   = 128
OUTPUT_SIZE   = 1
NUM_HID_LAYERS = 3
LEARNING_RATE = 0.001
NUM_EPOCHS    = 500
BATCH_SIZE    = 256

model     = HousePredictor(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HID_LAYERS).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ── Training ──────────────────────────────────────────────────────────────────

train_losses = []
test_losses  = []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    for i in range(0, len(X_train_t), BATCH_SIZE):
        X_batch = X_train_t[i:i + BATCH_SIZE]
        y_batch = y_train_t[i:i + BATCH_SIZE]

        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    avg_train_loss = epoch_loss / n_batches
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X_test_t), y_test_t).item()
    test_losses.append(test_loss)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  Train Loss: {avg_train_loss:.4f}  Test Loss: {test_loss:.4f}")


# ── Evaluation ────────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    y_pred = model(X_test_t).cpu().numpy()

y_test_np = y_test_t.cpu().numpy()

r2   = r2_score(y_test_np, y_pred)
rmse = np.sqrt(np.mean((y_pred - y_test_np) ** 2))

print(f"\nNN R² Score : {r2:.4f}")
print(f"NN RMSE     : {rmse:.4f}  (in $100k units, so ×$100k = ${rmse*100_000:,.0f})")
print(f"\nBaseline comparison:")
print(f"  Ridge          R²: 0.5758")
print(f"  Random Forest  R²: 0.8050")
print(f"  Neural Network R²: {r2:.4f}")


# ── Save weights ──────────────────────────────────────────────────────────────

torch.save(model.state_dict(), "calhouse_nn_weights.pth")
print("\nWeights saved to calhouse_nn_weights.pth")
