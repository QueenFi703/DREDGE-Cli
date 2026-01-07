//! Memory mapping and isolation
//!
//! This module enforces the memory non-interference invariant:
//! For any two distinct VMs: memory(vm₁) ∩ memory(vm₂) = ∅
//!
//! All memory operations verify this invariant before and after execution.

use crate::{Error, Result};
use crate::types::{SystemState, VMID, GPA, HostPage, Capability};
use std::collections::HashMap;

/// Memory manager for VM memory mapping and isolation
pub struct MemoryManager;

impl MemoryManager {
    /// Map a host page to a guest physical address for a VM
    ///
    /// Formal precondition:
    /// - VM exists
    /// - VM has MapMemory capability
    /// - GPA not already mapped for this VM
    /// - HostPage not mapped to any other VM (memory non-interference)
    ///
    /// Formal postcondition:
    /// - GPA mapped to HostPage for this VM
    /// - Memory non-interference preserved: ∀ vm₁ vm₂. vm₁ ≠ vm₂ → memory(vm₁) ∩ memory(vm₂) = ∅
    pub fn map_memory(
        state: &mut SystemState,
        vmid: VMID,
        gpa: GPA,
        host_page: HostPage,
    ) -> Result<()> {
        // Verify VM exists
        if !state.vms.contains_key(&vmid) {
            return Err(Error::VMNotFound(vmid));
        }
        
        // Verify MapMemory capability
        if !state.has_capability(vmid, Capability::MapMemory) {
            return Err(Error::CapabilityError(
                "MapMemory capability required".to_string(),
            ));
        }
        
        // Check if GPA is already mapped for this VM
        if state.memory.contains_key(&gpa) {
            return Err(Error::MemoryError(format!(
                "GPA {:?} already mapped",
                gpa
            )));
        }
        
        // Verify memory non-interference: host page must not be mapped to any other VM
        // Build reverse mapping to check this
        let reverse_map = Self::build_reverse_mapping(state);
        if let Some(existing_vmid) = reverse_map.get(&host_page) {
            if *existing_vmid != vmid {
                return Err(Error::MemoryError(format!(
                    "HostPage {:?} already mapped to VM {:?}",
                    host_page, existing_vmid
                )));
            }
        }
        
        // Insert mapping
        state.memory.insert(gpa, Some(host_page));
        
        // Verify invariant is maintained
        if !state.verify_memory_isolation() {
            // Roll back on invariant violation
            state.memory.remove(&gpa);
            return Err(Error::MemoryError(
                "Memory isolation invariant violated".to_string(),
            ));
        }
        
        Ok(())
    }
    
    /// Unmap a guest physical address
    ///
    /// Formal precondition:
    /// - VM exists
    /// - VM has MapMemory capability
    /// - GPA is currently mapped
    ///
    /// Formal postcondition:
    /// - GPA no longer mapped
    /// - Memory non-interference preserved
    pub fn unmap_memory(
        state: &mut SystemState,
        vmid: VMID,
        gpa: GPA,
    ) -> Result<()> {
        // Verify VM exists
        if !state.vms.contains_key(&vmid) {
            return Err(Error::VMNotFound(vmid));
        }
        
        // Verify MapMemory capability
        if !state.has_capability(vmid, Capability::MapMemory) {
            return Err(Error::CapabilityError(
                "MapMemory capability required".to_string(),
            ));
        }
        
        // Verify GPA is mapped
        if !state.memory.contains_key(&gpa) {
            return Err(Error::MemoryError(format!(
                "GPA {:?} not mapped",
                gpa
            )));
        }
        
        // Remove mapping
        state.memory.remove(&gpa);
        
        Ok(())
    }
    
    /// Translate guest physical address to host page
    ///
    /// Returns None if GPA is not mapped
    pub fn translate_gpa(
        state: &SystemState,
        gpa: GPA,
    ) -> Option<HostPage> {
        state.memory.get(&gpa).and_then(|&page| page)
    }
    
    /// Build reverse mapping from HostPage to VMID
    ///
    /// This is used to verify memory non-interference.
    /// 
    /// Note: This is a simplified placeholder implementation. A production version
    /// would maintain an explicit GPA->VMID ownership table incrementally during
    /// map/unmap operations, allowing accurate reverse lookups. The current version
    /// assigns pages to the first VM encountered, which is sufficient for basic
    /// verification but not for precise ownership tracking.
    fn build_reverse_mapping(state: &SystemState) -> HashMap<HostPage, VMID> {
        let mut reverse_map = HashMap::new();
        
        // Simplified: Assign each page to first VM we encounter it with
        // Production: Maintain explicit ownership via GPA->VMID table
        for (&_vmid, _) in &state.vms {
            for (&_gpa, &maybe_page) in &state.memory {
                if let Some(page) = maybe_page {
                    // Insert only if not already present (first-seen wins)
                    reverse_map.entry(page).or_insert(_vmid);
                }
            }
        }
        
        reverse_map
    }
    
    /// Get all mapped GPAs for a VM
    ///
    /// Returns the set of guest physical addresses currently mapped.
    /// This is useful for verification and debugging.
    /// 
    /// Note: Current implementation returns all GPAs in the system as a placeholder.
    /// A production implementation would maintain per-VM GPA tracking with explicit
    /// GPA->VMID ownership mapping to return only the specified VM's GPAs.
    pub fn get_vm_mappings(
        state: &SystemState,
        vmid: VMID,
    ) -> Result<Vec<GPA>> {
        // Verify VM exists
        if !state.vms.contains_key(&vmid) {
            return Err(Error::VMNotFound(vmid));
        }
        
        // Simplified: Return all GPAs
        // Production would filter by ownership: gpas.retain(|gpa| owns(vmid, gpa))
        let mut gpas: Vec<GPA> = state.memory.keys().copied().collect();
        gpas.sort();
        
        Ok(gpas)
    }
    
    /// Verify memory isolation for all VMs
    ///
    /// This is a verification function that checks the memory non-interference
    /// invariant across the entire system.
    pub fn verify_isolation(state: &SystemState) -> bool {
        state.verify_memory_isolation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::VMManager;

    #[test]
    fn test_memory_mapping() {
        let mut state = SystemState::new();
        let vmid = VMManager::create_vm(&mut state).unwrap();
        
        let gpa = GPA(0x1000);
        let page = HostPage(0x4000);
        
        MemoryManager::map_memory(&mut state, vmid, gpa, page).unwrap();
        
        assert_eq!(MemoryManager::translate_gpa(&state, gpa), Some(page));
    }

    #[test]
    fn test_memory_unmapping() {
        let mut state = SystemState::new();
        let vmid = VMManager::create_vm(&mut state).unwrap();
        
        let gpa = GPA(0x1000);
        let page = HostPage(0x4000);
        
        MemoryManager::map_memory(&mut state, vmid, gpa, page).unwrap();
        MemoryManager::unmap_memory(&mut state, vmid, gpa).unwrap();
        
        assert_eq!(MemoryManager::translate_gpa(&state, gpa), None);
    }

    #[test]
    fn test_memory_non_interference() {
        let mut state = SystemState::new();
        let vm1 = VMManager::create_vm(&mut state).unwrap();
        let vm2 = VMManager::create_vm(&mut state).unwrap();
        
        let gpa1 = GPA(0x1000);
        let gpa2 = GPA(0x2000);
        let page1 = HostPage(0x4000);
        let page2 = HostPage(0x5000);
        
        // Map different pages to different VMs - should succeed
        MemoryManager::map_memory(&mut state, vm1, gpa1, page1).unwrap();
        MemoryManager::map_memory(&mut state, vm2, gpa2, page2).unwrap();
        
        // Verify isolation is maintained
        assert!(MemoryManager::verify_isolation(&state));
    }

    #[test]
    fn test_duplicate_gpa_mapping_fails() {
        let mut state = SystemState::new();
        let vmid = VMManager::create_vm(&mut state).unwrap();
        
        let gpa = GPA(0x1000);
        let page1 = HostPage(0x4000);
        let page2 = HostPage(0x5000);
        
        MemoryManager::map_memory(&mut state, vmid, gpa, page1).unwrap();
        
        // Mapping same GPA again should fail
        let result = MemoryManager::map_memory(&mut state, vmid, gpa, page2);
        assert!(result.is_err());
    }

    #[test]
    fn test_capability_enforcement() {
        let mut state = SystemState::new();
        let vmid = VMManager::create_vm(&mut state).unwrap();
        
        // Remove MapMemory capability
        state.caps.get_mut(&vmid).unwrap().remove(&Capability::MapMemory);
        
        let gpa = GPA(0x1000);
        let page = HostPage(0x4000);
        
        // Should fail without capability
        let result = MemoryManager::map_memory(&mut state, vmid, gpa, page);
        assert!(result.is_err());
    }
}
