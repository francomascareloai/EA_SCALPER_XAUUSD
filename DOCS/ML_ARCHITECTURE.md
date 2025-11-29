# ML Architecture - EA_SCALPER_XAUUSD

## Model Specifications

| Model | Input | Output | Threshold |
|-------|-------|--------|-----------|
| Direction | (100, 15) | [P(bear), P(bull)] | >0.65 |
| Volatility | (50, 5) | ATR[5] | - |
| Fakeout | (20, 4) | [P(fake), P(real)] | <0.4 |

## Direction LSTM

```python
class DirectionLSTM(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return F.softmax(self.fc(out[:, -1, :]), dim=1)
```

## ONNX Export

```python
def export_to_onnx(model, dummy_input, path):
    model.eval()
    torch.onnx.export(model, dummy_input, path, opset_version=11,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
```

## MQL5 Integration

See: `DOCS/examples/onnx_brain.mqh`
