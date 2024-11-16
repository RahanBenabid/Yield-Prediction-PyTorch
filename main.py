import sys
sys.path.insert(0, './local_modules/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# Read and preprocess the data
df = pd.read_csv("./data/Crops_data.csv")

def create_features(df):
    # Create year-based features
    df['Year'] = pd.to_numeric(df['Year'])
    
    # Calculate area ratios and totals
    df['Total_Crop_Area'] = df[[col for col in df.columns if 'AREA' in col]].sum(axis=1)
    df['Rice_Area_Ratio'] = df['RICE AREA (1000 ha)'] / df['Total_Crop_Area']
    
    # Create production efficiency metrics
    df['Rice_Production_Efficiency'] = df['RICE PRODUCTION (1000 tons)'] / df['RICE AREA (1000 ha)']
    df['Overall_Production_Efficiency'] = df[[col for col in df.columns if 'PRODUCTION' in col]].sum(axis=1) / df['Total_Crop_Area']
    
    # Calculate moving averages for districts
    df['Prev_Year_Yield'] = df.groupby('Dist Name')['RICE YIELD (Kg per ha)'].shift(1)
    df['Yield_MA_3'] = df.groupby('Dist Name')['RICE YIELD (Kg per ha)'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    
    return df

# Create features
df = create_features(df)

# Select base features
base_features = [
    'RICE AREA (1000 ha)', 'WHEAT AREA (1000 ha)', 'MAIZE AREA (1000 ha)',
    'CHICKPEA AREA (1000 ha)', 'GROUNDNUT AREA (1000 ha)', 
    'SUGARCANE AREA (1000 ha)', 'COTTON AREA (1000 ha)',
    'Total_Crop_Area', 'Rice_Area_Ratio', 'Rice_Production_Efficiency',
    'Overall_Production_Efficiency', 'Prev_Year_Yield', 'Yield_MA_3'
]

# Remove rows with missing values
df = df.dropna(subset=base_features + ['RICE YIELD (Kg per ha)'])

# Prepare features and target
X = df[base_features].values
y = df['RICE YIELD (Kg per ha)'].values

# Feature selection
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X, y)
selected_features_mask = selector.get_support()
selected_features = [feat for feat, selected in zip(base_features, selected_features_mask) if selected]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Use RobustScaler for better outlier handling
feature_scaler = RobustScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)

y_scaler = RobustScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class EnhancedCropYieldModel(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedCropYieldModel, self).__init__()
        # First layer
        self.linear1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        # Second layer
        self.linear2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # Third layer
        self.linear3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layer
        self.linear4 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.linear1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.linear2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.linear3(x))))
        return self.linear4(x)
    
# Create model instance
model = EnhancedCropYieldModel(input_dim=X_selected.shape[1])

# Loss function and optimizer
criterion = nn.HuberLoss(delta=1.0)  # More robust to outliers than MSE
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)

# Training loop
epochs = 200
batch_size = 32
training_losses = []
validation_losses = []
best_val_loss = float('inf')
patience = 100
patience_counter = 0

print("Training the model...")
print(f"Selected features: {selected_features}")

for epoch in range(epochs):
    model.train()
    epoch_losses = []
    
    # Mini-batch training
    indices = torch.randperm(X_train_tensor.shape[0])
    for i in range(0, X_train_tensor.shape[0], batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_X = X_train_tensor[batch_indices]
        batch_y = y_train_tensor[batch_indices]
        
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        epoch_losses.append(loss.item())
        
    # Calculate training and validation loss
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(X_train_tensor), y_train_tensor)
        val_loss = criterion(model(X_test_tensor), y_test_tensor)
        training_losses.append(train_loss.item())
        validation_losses.append(val_loss.item())
        
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
        
# Model evaluation
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor).numpy()
    
# Transform predictions back to original scale
y_pred_original = y_scaler.inverse_transform(y_pred_test)
y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)
mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

print(f'\nModel Performance:')
print(f'Mean Squared Error: {mse:,.2f}')
print(f'Root Mean Squared Error: {rmse:,.2f}')
print(f'R-squared: {r2:.4f}')
print(f'Mean Absolute Percentage Error: {mape:.2f}%')

# Plotting results
plt.figure(figsize=(15, 5))

# Plot 1: Predictions vs Actual
plt.subplot(1, 2, 1)
plt.scatter(y_test_original, y_pred_original, alpha=0.5, color='blue', label='Predictions')
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 
         'r--', label='Perfect Prediction')
plt.xlabel("Actual Rice Yield (Kg per ha)")
plt.ylabel("Predicted Rice Yield (Kg per ha)")
plt.title(f"Actual vs Predicted Rice Yield\nR² = {r2:.4f}")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Training History
plt.subplot(1, 2, 2)
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()