import opendssdirect as dss
import numpy as np
import pandas as pd
import random
import os
import time
from datetime import datetime

def create_fault_dataset(
    dss_file_path, 
    output_dir, 
    feeder_name,
    n_faults_per_bus=48,  # Default to 48 to be divisible by 3 fault types
    fault_types=["SPG", "PP", "DPG"],
    impedance_range=(0.05, 0.1)
):
    """
    Create a fault detection dataset for a given OpenDSS model with incremental file writing
    to minimize memory usage.
    
    Parameters:
    -----------
    dss_file_path : str
        Path to the main OpenDSS file for the model
    output_dir : str
        Directory to save output files
    feeder_name : str
        Name of the feeder (used for file naming)
    n_faults_per_bus : int
        Number of faults to generate per 3-phase bus
    fault_types : list
        Types of faults to generate
    impedance_range : tuple
        Range of fault impedances (min, max)
    """
    # Start timing
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    features_file = os.path.join(output_dir, f"{feeder_name}_features.csv")
    labels_file = os.path.join(output_dir, f"{feeder_name}_labels.csv")
    metadata_file = os.path.join(output_dir, f"{feeder_name}_metadata.txt")
    
    # Step 1: Load OpenDSS model
    print(f"Loading {feeder_name} model...")
    dss.Text.Command("Clear")
    dss.Text.Command(f"Redirect {dss_file_path}")
    dss.Text.Command("Solve")
    
    # Step 2: Identify three-phase buses
    all_buses = dss.Circuit.AllBusNames()
    three_phase_buses = []

    for bus in all_buses:
        dss.Circuit.SetActiveBus(bus)
        voltages = dss.Bus.puVmagAngle()
        num_phases = len(voltages) // 2
        if num_phases == 3:
            three_phase_buses.append(bus)

    if not three_phase_buses:
        raise ValueError(f"No three-phase buses found in {feeder_name}!")
    
    print(f"Found {len(three_phase_buses)} three-phase buses in {feeder_name}")
    
    # Calculate faults per type to ensure exactly n_faults_per_bus total faults per bus
    faults_per_type = n_faults_per_bus // len(fault_types)
    if faults_per_type * len(fault_types) != n_faults_per_bus:
        adjusted_faults = faults_per_type * len(fault_types)
        print(f"Warning: {n_faults_per_bus} faults per bus is not divisible by {len(fault_types)} fault types.")
        print(f"Adjusting to {adjusted_faults} faults per bus.")
        n_faults_per_bus = adjusted_faults
    
    # Initialize CSV files (create new or truncate existing)
    with open(features_file, 'w') as f_features, open(labels_file, 'w') as f_labels:
        pass  # Just create/truncate the files
    
    # Track total samples and sample fault details for metadata
    total_samples = 0
    sample_fault_info = []  # Store only a few fault descriptions for metadata
    
    # Step 3: Generate fault cases, process one bus at a time
    for bus_idx, bus in enumerate(three_phase_buses):
        print(f"Processing bus {bus_idx+1}/{len(three_phase_buses)}: {bus}")
        
        # Temporary storage for current bus data
        bus_features_data = []
        bus_labels_data = []
        
        for fault_type in fault_types:
            for i in range(faults_per_type):
                # Random fault impedance
                fault_impedance = round(random.uniform(*impedance_range), 4)

                # Apply randomized load fluctuation
                dss.Text.Command("ClearAll")  # Reset system state
                dss.Text.Command(f"Redirect {dss_file_path}")  # Reload system
                dss.Text.Command("Solve")
                
                # Apply fault
                fault_description = ""
                if fault_type == "SPG":   # Single-phase-to-ground
                    phase = random.choice([".1", ".2", ".3"])
                    dss.Text.Command(f"New Fault.F1 Bus1={bus}{phase} Phases=1 r={fault_impedance}")
                    fault_description = f"SPG fault at {bus}{phase}, r={fault_impedance} ohm"
                elif fault_type == "PP":  # Phase-to-phase
                    phases = random.choice([(1,2), (2,3), (1,3)])
                    dss.Text.Command(f"New Fault.F1 Bus1={bus}.{phases[0]} Bus2={bus}.{phases[1]} Phases=2 r={fault_impedance}")
                    fault_description = f"PP fault at {bus}.{phases[0]}-{bus}.{phases[1]}, r={fault_impedance} ohm"
                elif fault_type == "DPG": # Double-phase-to-ground
                    phases = random.choice([".1.2", ".2.3", ".1.3"])
                    dss.Text.Command(f"New Fault.F1 Bus1={bus}{phases} Phases=2 r={fault_impedance}")
                    fault_description = f"DPG fault at {bus}{phases}, r={fault_impedance} ohm"

                dss.Text.Command("Solve")

                # Collect per-unit voltage data
                voltages = []
                for b in all_buses:
                    dss.Circuit.SetActiveBus(b)
                    bus_voltages = dss.Bus.puVmagAngle()
                    voltages.extend(bus_voltages[::2])  # Extract real parts only
                
                bus_features_data.append(voltages)
                bus_labels_data.append(bus)
                
                # Store a sample of fault descriptions for metadata
                if total_samples < 10:
                    sample_fault_info.append(fault_description)
                
                total_samples += 1
                
        # Append this bus's data to CSV files
        pd.DataFrame(bus_features_data).to_csv(features_file, index=False, header=False, mode='a')
        pd.DataFrame(bus_labels_data).to_csv(labels_file, index=False, header=False, mode='a')
        
        # Free up memory
        bus_features_data = []
        bus_labels_data = []
        
        # Report progress periodically
        if (bus_idx + 1) % 10 == 0 or bus_idx == len(three_phase_buses) - 1:
            print(f"Progress: {bus_idx + 1}/{len(three_phase_buses)} buses processed "
                  f"({((bus_idx + 1) / len(three_phase_buses) * 100):.1f}%)")
    
    # Get feature dimension by checking a single fault case
    # Reset the system
    dss.Text.Command("ClearAll")
    dss.Text.Command(f"Redirect {dss_file_path}")
    dss.Text.Command("Solve")
    
    # Apply a simple fault to get feature dimension
    test_bus = three_phase_buses[0]
    dss.Text.Command(f"New Fault.F1 Bus1={test_bus}.1 Phases=1 r=0.1")
    dss.Text.Command("Solve")
    
    feature_dimension = 0
    for b in all_buses:
        dss.Circuit.SetActiveBus(b)
        bus_voltages = dss.Bus.puVmagAngle()
        feature_dimension += len(bus_voltages) // 2
    
    # Save metadata - With utf-8 encoding and "ohm" instead of Omega
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset for {feeder_name} created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"OpenDSS model: {dss_file_path}\n")
        f.write(f"Total number of buses: {len(all_buses)}\n")
        f.write(f"Number of three-phase buses: {len(three_phase_buses)}\n")
        f.write(f"Fault types: {', '.join(fault_types)}\n")
        f.write(f"Faults per bus: {n_faults_per_bus}\n")
        f.write(f"Impedance range: {impedance_range[0]}-{impedance_range[1]} ohm\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Feature dimension: {feature_dimension}\n")
        f.write(f"Generation time: {elapsed_time:.2f} seconds\n\n")
        f.write("Three-phase buses:\n")
        for bus in three_phase_buses[:20]:  # Limit to 20 buses to avoid huge metadata files
            f.write(f"- {bus}\n")
        if len(three_phase_buses) > 20:
            f.write(f"... and {len(three_phase_buses) - 20} more buses\n")
        
        f.write("\nSample fault details (first 10):\n")
        for i, fault_info in enumerate(sample_fault_info):
            f.write(f"{i+1}. {fault_info}\n")
    
    print(f"Dataset generation for {feeder_name} completed!")
    print(f"Created {total_samples} samples in {elapsed_time:.2f} seconds")
    print(f"Output saved to {output_dir}")
    
    return total_samples

# Define paths and parameters for each test feeder
feeders = [
    {
        "name": "IEEE13",
        "dss_file": "13Bus/IEEE13Nodeckt.dss",
        "output_dir": "13data"
    },
    {
        "name": "IEEE34",
        "dss_file": "34Bus/ieee34Mod1.dss",
        "output_dir": "34data"
    },
    {
        "name": "IEEE37",
        "dss_file": "37Bus/ieee37.dss",
        "output_dir": "37data"
    },
    {
        "name": "IEEE123",
        "dss_file": "123Bus/IEEE123Master.dss",
        "output_dir": "123data"
    },
    {
        "name": "EuropeanLV",
        "dss_file": "EuropeanLV/Master.dss",
        "output_dir": "EuropeanLVdata"
    }
]

if __name__ == "__main__":
    print("IEEE Test Feeders Dataset Generator")
    print("==================================")
    
    # Get user input for which feeders to process
    print("\nAvailable feeders:")
    for i, feeder in enumerate(feeders):
        print(f"{i+1}. {feeder['name']}")
    print(f"{len(feeders)+1}. All feeders")
    
    choice = input("\nEnter the number(s) of the feeder(s) to process (comma-separated for multiple): ")
    
    selected_indices = []
    if choice.strip() == str(len(feeders)+1):
        selected_indices = list(range(len(feeders)))
    else:
        for idx in choice.split(','):
            try:
                idx = int(idx.strip()) - 1
                if 0 <= idx < len(feeders):
                    selected_indices.append(idx)
                else:
                    print(f"Invalid index {idx+1}, skipping.")
            except ValueError:
                print(f"Invalid input '{idx}', skipping.")
    
    if not selected_indices:
        print("No valid feeders selected. Exiting.")
        exit()
    
    # Allow user to customize faults per bus for each feeder
    for idx in selected_indices:
        feeder = feeders[idx]
        print(f"\n=== Generating dataset for {feeder['name']} ===\n")
        
        faults_per_bus = 48  # Default value
        
        # Ask for custom number of faults only for EuropeanLV
        if feeder["name"] == "EuropeanLV":
            try:
                custom_faults = input(f"Enter number of faults per bus for {feeder['name']} (default is 48, must be divisible by 3): ")
                if custom_faults.strip():
                    faults_per_bus = int(custom_faults)
                    if faults_per_bus % 3 != 0:
                        print(f"Warning: {faults_per_bus} is not divisible by 3. Adjusting to {faults_per_bus - (faults_per_bus % 3)}")
                        faults_per_bus = faults_per_bus - (faults_per_bus % 3)
            except ValueError:
                print("Invalid input, using default value of 48 faults per bus.")
                faults_per_bus = 48
        
        try:
            n_samples = create_fault_dataset(
                feeder["dss_file"],
                feeder["output_dir"],
                feeder["name"],
                n_faults_per_bus=faults_per_bus,
                fault_types=["SPG", "PP", "DPG"],
                impedance_range=(0.05, 0.1)
            )
            print(f"Successfully created {n_samples} samples for {feeder['name']}")
        except Exception as e:
            print(f"Error generating dataset for {feeder['name']}: {str(e)}")
    
    print("\nAll requested datasets have been generated.")