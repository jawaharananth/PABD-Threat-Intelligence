import numpy as np
import pandas as pd

np.random.seed(42)

def generate_normal_traffic(n_samples=5000):
    data = {
        'connection_frequency': np.random.normal(5, 1, n_samples),
        'port_entropy': np.random.normal(0.3, 0.05, n_samples),
        'failed_auth_rate': np.random.normal(0.02, 0.005, n_samples),
        'packet_size_dist': np.random.normal(512, 50, n_samples),
        'protocol_ratio': np.random.normal(0.95, 0.02, n_samples),
        'burst_activity': np.random.normal(0.12, 0.02, n_samples),
        'label': 0,
        'attack_type': 'Normal'
    }
    return pd.DataFrame(data)

def generate_reconnaissance(n_samples=1000):
    data = {
        'connection_frequency': np.random.normal(25, 5, n_samples),
        'port_entropy': np.random.normal(0.85, 0.08, n_samples),
        'failed_auth_rate': np.random.normal(0.15, 0.03, n_samples),
        'packet_size_dist': np.random.normal(128, 30, n_samples),
        'protocol_ratio': np.random.normal(0.60, 0.08, n_samples),
        'burst_activity': np.random.normal(0.75, 0.10, n_samples),
        'label': 1,
        'attack_type': 'Reconnaissance'
    }
    return pd.DataFrame(data)

def generate_brute_force(n_samples=1000):
    data = {
        'connection_frequency': np.random.normal(18, 4, n_samples),
        'port_entropy': np.random.normal(0.45, 0.06, n_samples),
        'failed_auth_rate': np.random.normal(0.72, 0.08, n_samples),
        'packet_size_dist': np.random.normal(256, 40, n_samples),
        'protocol_ratio': np.random.normal(0.75, 0.06, n_samples),
        'burst_activity': np.random.normal(0.85, 0.08, n_samples),
        'label': 2,
        'attack_type': 'Brute Force'
    }
    return pd.DataFrame(data)

def generate_lateral_movement(n_samples=1000):
    data = {
        'connection_frequency': np.random.normal(12, 3, n_samples),
        'port_entropy': np.random.normal(0.55, 0.07, n_samples),
        'failed_auth_rate': np.random.normal(0.25, 0.04, n_samples),
        'packet_size_dist': np.random.normal(768, 60, n_samples),
        'protocol_ratio': np.random.normal(0.45, 0.07, n_samples),
        'burst_activity': np.random.normal(0.55, 0.08, n_samples),
        'label': 3,
        'attack_type': 'Lateral Movement'
    }
    return pd.DataFrame(data)

def generate_dos_preparation(n_samples=1000):
    data = {
        'connection_frequency': np.random.normal(45, 8, n_samples),
        'port_entropy': np.random.normal(0.25, 0.04, n_samples),
        'failed_auth_rate': np.random.normal(0.05, 0.01, n_samples),
        'packet_size_dist': np.random.normal(64, 10, n_samples),
        'protocol_ratio': np.random.normal(0.55, 0.05, n_samples),
        'burst_activity': np.random.normal(0.95, 0.03, n_samples),
        'label': 4,
        'attack_type': 'DoS Preparation'
    }
    return pd.DataFrame(data)

def generate_pre_attack_sequence(attack_type='Reconnaissance', n_steps=50):
    sequence = []
    if attack_type == 'Reconnaissance':
        for i in range(n_steps):
            progress = i / n_steps
            sequence.append({
                'connection_frequency': 5 + (20 * progress) + np.random.normal(0, 0.5),
                'port_entropy': 0.3 + (0.55 * progress) + np.random.normal(0, 0.02),
                'failed_auth_rate': 0.02 + (0.13 * progress) + np.random.normal(0, 0.005),
                'packet_size_dist': 512 - (384 * progress) + np.random.normal(0, 10),
                'protocol_ratio': 0.95 - (0.35 * progress) + np.random.normal(0, 0.02),
                'burst_activity': 0.12 + (0.63 * progress) + np.random.normal(0, 0.02),
                'step': i,
                'progress': progress,
                'attack_type': attack_type
            })
    elif attack_type == 'Brute Force':
        for i in range(n_steps):
            progress = i / n_steps
            sequence.append({
                'connection_frequency': 5 + (13 * progress) + np.random.normal(0, 0.5),
                'port_entropy': 0.3 + (0.15 * progress) + np.random.normal(0, 0.02),
                'failed_auth_rate': 0.02 + (0.70 * progress) + np.random.normal(0, 0.01),
                'packet_size_dist': 512 - (256 * progress) + np.random.normal(0, 10),
                'protocol_ratio': 0.95 - (0.20 * progress) + np.random.normal(0, 0.02),
                'burst_activity': 0.12 + (0.73 * progress) + np.random.normal(0, 0.02),
                'step': i,
                'progress': progress,
                'attack_type': attack_type
            })
    elif attack_type == 'Lateral Movement':
        for i in range(n_steps):
            progress = i / n_steps
            sequence.append({
                'connection_frequency': 5 + (7 * progress) + np.random.normal(0, 0.5),
                'port_entropy': 0.3 + (0.25 * progress) + np.random.normal(0, 0.02),
                'failed_auth_rate': 0.02 + (0.23 * progress) + np.random.normal(0, 0.005),
                'packet_size_dist': 512 + (256 * progress) + np.random.normal(0, 10),
                'protocol_ratio': 0.95 - (0.50 * progress) + np.random.normal(0, 0.02),
                'burst_activity': 0.12 + (0.43 * progress) + np.random.normal(0, 0.02),
                'step': i,
                'progress': progress,
                'attack_type': attack_type
            })
    return pd.DataFrame(sequence)

def generate_full_dataset():
    normal = generate_normal_traffic(5000)
    recon = generate_reconnaissance(1000)
    brute = generate_brute_force(1000)
    lateral = generate_lateral_movement(1000)
    dos = generate_dos_preparation(1000)
    df = pd.concat([normal, recon, brute, lateral, dos], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df['connection_frequency'] = df['connection_frequency'].clip(0, 100)
    df['port_entropy'] = df['port_entropy'].clip(0, 1)
    df['failed_auth_rate'] = df['failed_auth_rate'].clip(0, 1)
    df['packet_size_dist'] = df['packet_size_dist'].clip(64, 1500)
    df['protocol_ratio'] = df['protocol_ratio'].clip(0, 1)
    df['burst_activity'] = df['burst_activity'].clip(0, 1)
    return df

if __name__ == "__main__":
    df = generate_full_dataset()
    df.to_csv('network_data.csv', index=False)
    print(f"Dataset generated: {len(df)} samples")
    print(df['attack_type'].value_counts())
    print("Saved to network_data.csv")