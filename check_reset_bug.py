"""
Sanity check script to verify whether T2D custom parameter modifications
(Vmx, kp3) survive simglucose patient.reset() calls.
"""

from run_selective_warmstart_simglucose_hybrid import build_calibrated_t2d_patient

def verify_reset_persistence():
    patient = build_calibrated_t2d_patient()
    
    vmx_modified = patient._params['Vmx']
    kp3_modified = patient._params['kp3']
    
    print("=" * 55)
    print("   SIMGLUCOSE PATIENT RESET PARAMETER SURVIVAL TEST   ")
    print("=" * 55)
    print(f"[BEFORE RESET] Vmx: {vmx_modified:.8f} | kp3: {kp3_modified:.8f}")
    
    # Execute patient reset
    patient.reset()
    
    vmx_after = patient._params['Vmx']
    kp3_after = patient._params['kp3']
    
    print(f"[AFTER RESET]  Vmx: {vmx_after:.8f} | kp3: {kp3_after:.8f}")
    print("=" * 55)
    
    if vmx_modified == vmx_after and kp3_modified == kp3_after:
        print("✓ SUCCESS: Parameters survived patient.reset() intact!")
    else:
        print("❌ CRITICAL BUG DETECTED: patient.reset() wiped custom parameters!")
        print("   Stock parameters were restored during reset.")

if __name__ == "__main__":
    verify_reset_persistence()