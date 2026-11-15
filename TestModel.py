import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

seed = 42
epoci = 5
batch = 64
torch.set_num_threads(1)

X, y = load_digits(return_X_y=True)
X = X / 16.0

X, X_test, y, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=seed
)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=seed
)

X_train, X_val, X_test = [
    torch.tensor(date, dtype=torch.float32) for date in (X_train, X_val, X_test)
]
y_train, y_val, y_test = [
    torch.tensor(etichete, dtype=torch.long) for etichete in (y_train, y_val, y_test)
]


def antreneaza(eta, beta):
    torch.manual_seed(seed)
    retea = nn.Sequential(nn.Linear(64, 64), nn.Sigmoid(), nn.Linear(64, 10))
    optimizator = torch.optim.SGD(retea.parameters(), lr=eta, momentum=beta)
    pierdere = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    for _ in range(epoci):
        ordine = torch.randperm(len(y_train), generator=generator)
        for indici in ordine.split(batch):
            optimizator.zero_grad()
            loss = pierdere(retea(X_train[indici]), y_train[indici])
            loss.backward()
            optimizator.step()

    return retea


def acuratete(retea, X, y):
    retea.eval()
    with torch.no_grad():
        predictii = retea(X).argmax(dim=1)
        return (predictii == y).float().mean().item()


eta_valori = [0.01, 0.03, 0.1, 0.3]
beta_valori = [0.8, 0.9, 0.95]
scoruri = np.zeros((len(beta_valori), len(eta_valori)))

for i, beta in enumerate(beta_valori):
    for j, eta in enumerate(eta_valori):
        retea = antreneaza(eta, beta)
        scoruri[i, j] = acuratete(retea, X_val, y_val)
        print(f"eta={eta:g}, beta={beta:g}, validare={scoruri[i, j]:.2%}")

i, j = np.unravel_index(scoruri.argmax(), scoruri.shape)
eta_bun, beta_bun = eta_valori[j], beta_valori[i]
retea = antreneaza(eta_bun, beta_bun)
print(f"\nPerechea aleasa: eta={eta_bun:g}, beta={beta_bun:g}")
print(f"Acuratete test: {acuratete(retea, X_test, y_test):.2%}")

# Grafic 3D pentru perechile fixe
E, B = np.meshgrid(np.log10(eta_valori), beta_valori)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(E, B, scoruri * 100, cmap="viridis", alpha=0.8)
ax.scatter(E, B, scoruri * 100, color="black")

ax.set_xticks(np.log10(eta_valori), [str(eta) for eta in eta_valori])
ax.set_yticks(beta_valori)
ax.set_xlabel("Eta (scara log)")
ax.set_ylabel("Beta")
ax.set_zlabel("Acuratete validare (%)", labelpad=8)
ax.set_title("Digits - momentum fix")
ax.view_init(elev=27, azim=-126)
fig.subplots_adjust(left=0.09, right=0.90, bottom=0.10, top=0.89)
fig.savefig("acuratete_3d.png", dpi=160)
plt.show()
