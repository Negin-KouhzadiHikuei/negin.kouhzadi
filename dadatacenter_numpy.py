import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

server_racks = np.random.uniform(low=1.5, high=5.0, size=(3, 4))

server_racks = np.round(server_racks, 1)

print("--- armania datacenters- automatic---" )
print(server_racks)
print(f"array shape: {server_racks.shape} → {server_racks.shape[0]} shelf: {server_racks.shape[1]} server")


rack_means = np.mean(server_racks, axis=1)
print(f"average consumption per shelf: {rack_means}")

max_power = np.max(server_racks)
print(f"most consumed: {max_power}")


threshold = np.mean(server_racks) - 0.5
print(f"\n optimization limit(smart): {round(threshold, 2)} kilowatt")

low_power_mask = server_racks < threshold
server_racks[low_power_mask] = round(threshold, 1)

print("\n---after smart optimization---")
print(server_racks)


rack_labels = ['rack a', 'rack b', 'rack c']

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(rack_labels))

width = 0.2

ax.bar(x - 1.5*width, server_racks[: , 0], width, label='server 1', color='steelblue')
ax.bar(x - 0.5*width, server_racks[: , 1], width, label='server 2', color='orange')
ax.bar(x + 0.5*width, server_racks[: , 2], width, label='server 3', color='green')
ax.bar(x + 1.5*width, server_racks[: , 3], width, label='server 4', color='red')

ax.axhline(y=4.5, color='darkred', linestyle='--', linewidth=1.5 , label='critical limit')

ax.axhline(y=threshold, color='blue', linestyle=':', linewidth=1.5, label=f'optimal threshold')


ax.set_xlabel('server rack')
ax.set_ylabel('power usage(kw)')
ax.set_title('armania datacenter - power optimization')
ax.set_xticks(x)
ax.set_xticklabels(rack_labels)
ax.legend()


plt.tight_layout()

plt.savefig('baseline_underfitting.png', dpi=300)
plt.show()