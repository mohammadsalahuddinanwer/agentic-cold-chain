from data_generation.generator import ColdChainGenerator

def main():
    gen = ColdChainGenerator(seed=42)
    shipments = gen.generate_dataset(10)
    
    print("Checking for anomalies in 10 generated shipments:\n")
    for s in shipments:
        print(f"Shipment {s.shipment_id}: {len(s.events)} anomalies")

if __name__ == "__main__":
    main()