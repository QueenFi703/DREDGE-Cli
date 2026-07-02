#!/usr/bin/env python3
"""
DREDGE Studio Integration with Gordon
Adds DREDGE capabilities to Gordon's toolkit
"""

import os
import sys
from pathlib import Path

# Add DREDGE to path
try:
    import dredge
    DREDGE_AVAILABLE = True
    DREDGE_PATH = Path(dredge.__file__).parent
except ImportError:
    DREDGE_AVAILABLE = False
    DREDGE_PATH = None

# Gordon integration
GORDON_VERSION = "1.0.0"
DREDGE_INTEGRATION_VERSION = "1.0.0"


class GordonDREDGEIntegration:
    """Bridge between Gordon and DREDGE Studio"""
    
    def __init__(self):
        self.dredge_available = DREDGE_AVAILABLE
        self.dredge_version = dredge.__version__ if DREDGE_AVAILABLE else "NOT INSTALLED"
        self.modules = self._load_modules()
    
    def _load_modules(self):
        """Load all DREDGE modules"""
        if not DREDGE_AVAILABLE:
            return {}
        
        modules = {}
        try:
            from dredge import (
                advanced_features,
                dependabot_alerts,
                server,
                string_theory,
                monitoring,
                orchestration,
                cache,
                auth,
                cli,
            )
            
            modules = {
                'advanced_features': advanced_features,
                'dependabot_alerts': dependabot_alerts,
                'server': server,
                'string_theory': string_theory,
                'monitoring': monitoring,
                'orchestration': orchestration,
                'cache': cache,
                'auth': auth,
                'cli': cli,
            }
        except Exception as e:
            print(f"Error loading DREDGE modules: {e}")
        
        return modules
    
    def get_status(self):
        """Get DREDGE integration status"""
        return {
            "dredge_available": self.dredge_available,
            "dredge_version": self.dredge_version,
            "dredge_path": str(DREDGE_PATH),
            "modules_loaded": len(self.modules),
            "integration_version": DREDGE_INTEGRATION_VERSION,
        }
    
    def analyze_vulnerability(self, package, severity):
        """Use FiBot to analyze a vulnerability"""
        if not self.dredge_available:
            return {"error": "DREDGE not available"}
        
        try:
            from dredge.dependabot_alerts import FiBotSecurityAnalyzer
            fibot = FiBotSecurityAnalyzer()
            return fibot.analyze_vulnerability(package, severity)
        except Exception as e:
            return {"error": str(e)}
    
    def compute_string_spectrum(self, dimensions=10, max_modes=64):
        """Compute string vibrational spectrum"""
        if not self.dredge_available:
            return {"error": "DREDGE not available"}
        
        try:
            from dredge.string_theory import StringTheory
            st = StringTheory(dimensions=dimensions, max_modes=max_modes)
            return st.compute_spectrum()
        except Exception as e:
            return {"error": str(e)}
    
    def get_monitoring_metrics(self):
        """Get system monitoring metrics"""
        if not self.dredge_available:
            return {"error": "DREDGE not available"}
        
        try:
            from dredge.monitoring import metrics_collector
            return metrics_collector.get_metrics()
        except Exception as e:
            return {"error": str(e)}


# Initialize integration
dredge_integration = GordonDREDGEIntegration()


def get_dredge_info():
    """Get DREDGE information"""
    return {
        "name": "DREDGE Studio",
        "version": dredge_integration.dredge_version,
        "status": "Available" if dredge_integration.dredge_available else "Not Installed",
        "integration_version": DREDGE_INTEGRATION_VERSION,
        "modules": list(dredge_integration.modules.keys()),
        "capabilities": [
            "FiBot Security Analysis",
            "Dependabot Alert Management",
            "String Theory Computation",
            "Monitoring & Metrics",
            "Workflow Orchestration",
            "Web Server",
        ]
    }


def start_dredge_server(host="127.0.0.1", port=8000):
    """Start DREDGE web server"""
    if not dredge_integration.dredge_available:
        return {"error": "DREDGE not available"}
    
    try:
        from dredge.server import app
        print(f"Starting DREDGE server on {host}:{port}")
        app.run(host=host, port=port, debug=True)
    except Exception as e:
        return {"error": str(e)}


def fibot_security_analysis(package, severity, cve_details=None):
    """Run FiBot security analysis"""
    return dredge_integration.analyze_vulnerability(package, severity)


def string_theory_compute(dimensions=10, max_modes=64):
    """Compute string theory spectrum"""
    return dredge_integration.compute_string_spectrum(dimensions, max_modes)


def get_monitoring_stats():
    """Get monitoring statistics"""
    return dredge_integration.get_monitoring_metrics()


# Export for Gordon
__all__ = [
    'GordonDREDGEIntegration',
    'dredge_integration',
    'get_dredge_info',
    'start_dredge_server',
    'fibot_security_analysis',
    'string_theory_compute',
    'get_monitoring_stats',
]


if __name__ == "__main__":
    print("="*70)
    print("DREDGE Studio - Gordon Integration")
    print("="*70)
    print()
    print(get_dredge_info())
