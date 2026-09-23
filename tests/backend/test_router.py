import pytest
from intelligence.contracts import (
    IntelligenceRequest,
    ModelProfile,
    ModelCapability,
    TaskComplexity
)
from intelligence.router import ModelRouter
from intelligence.fabric import ProviderFabric
from providers.nvidia_provider import NVIDIAProvider

def test_router_selects_local_for_simple_tasks():
    fabric = ProviderFabric()
    local_4b = NVIDIAProvider(
        model_name="Nemotron-Mini-4B", local=True, api_key="dummy", base_url="mock"
    )
    remote_ultra = NVIDIAProvider(
        model_name="nvidia/nemotron-ultra", local=False, api_key="dummy", base_url="mock"
    )
    fabric.register(local_4b)
    fabric.register(remote_ultra)
    
    router = ModelRouter(fabric=fabric)
    
    # Mock VRAM estimation to 6.0GB
    router._cached_vram = 6.0
    router._vram_cache_time = 9999999999 # never expire
    
    req = IntelligenceRequest(prompt="Hello", complexity=TaskComplexity.SIMPLE)
    provider = router.route(req)
    
    assert provider.profile.local == True
    assert provider.profile.name == "Nemotron-Mini-4B"

def test_router_selects_remote_for_complex_tasks():
    fabric = ProviderFabric()
    local_4b = NVIDIAProvider(
        model_name="Nemotron-Mini-4B", local=True, api_key="dummy", base_url="mock"
    )
    remote_ultra = NVIDIAProvider(
        model_name="nvidia/nemotron-ultra", local=False, api_key="dummy", base_url="mock", cost_tier="frontier"
    )
    fabric.register(local_4b)
    fabric.register(remote_ultra)
    
    router = ModelRouter(fabric=fabric)
    router._cached_vram = 6.0
    
    req = IntelligenceRequest(prompt="Prove P=NP", complexity=TaskComplexity.COMPLEX)
    provider = router.route(req)
    
    assert provider.profile.local == False
    assert provider.profile.name == "nvidia/nemotron-ultra"

def test_router_selects_omni_for_agentic_tasks():
    fabric = ProviderFabric()
    local_4b = NVIDIAProvider(
        model_name="Nemotron-Mini-4B", local=True, api_key="dummy", base_url="mock"
    )
    remote_ultra = NVIDIAProvider(
        model_name="nvidia/nemotron-ultra", local=False, api_key="dummy", base_url="mock", cost_tier="frontier"
    )
    remote_omni = NVIDIAProvider(
        model_name="nvidia/nemotron-omni-30b", local=False, api_key="dummy", base_url="mock", cost_tier="high"
    )
    fabric.register(local_4b)
    fabric.register(remote_ultra)
    fabric.register(remote_omni)
    
    router = ModelRouter(fabric=fabric)
    router._cached_vram = 6.0
    
    req = IntelligenceRequest(prompt="Look at this image", task="multimodal")
    provider = router.route(req)
    
    assert "omni" in provider.profile.name.lower()

def test_router_forces_remote_if_context_large():
    fabric = ProviderFabric()
    # local provider has 4096 context
    local_4b = NVIDIAProvider(
        model_name="Nemotron-Mini-4B", local=True, api_key="dummy", base_url="mock", context_window=4096
    )
    remote_ultra = NVIDIAProvider(
        model_name="nvidia/nemotron-ultra", local=False, api_key="dummy", base_url="mock", cost_tier="frontier"
    )
    fabric.register(local_4b)
    fabric.register(remote_ultra)
    
    router = ModelRouter(fabric=fabric)
    router._cached_vram = 6.0
    
    # 4 characters = 1 token (roughly). We need > 4096 * 0.8 tokens = 3276 tokens = 13104 characters.
    long_prompt = "A" * 15000 
    
    req = IntelligenceRequest(prompt=long_prompt, complexity=TaskComplexity.SIMPLE)
    provider = router.route(req)
    
    # Even though simple, it should route to remote due to context overflow
    assert provider.profile.local == False
    assert provider.profile.name == "nvidia/nemotron-ultra"

def test_router_forces_remote_if_vram_low():
    fabric = ProviderFabric()
    local_4b = NVIDIAProvider(
        model_name="Nemotron-Mini-4B", local=True, api_key="dummy", base_url="mock"
    )
    remote_ultra = NVIDIAProvider(
        model_name="nvidia/nemotron-ultra", local=False, api_key="dummy", base_url="mock", cost_tier="frontier"
    )
    fabric.register(local_4b)
    fabric.register(remote_ultra)
    
    router = ModelRouter(fabric=fabric)
    # Simulate low VRAM
    router._cached_vram = 1.0 
    router._vram_cache_time = 9999999999
    
    req = IntelligenceRequest(prompt="Hello", complexity=TaskComplexity.SIMPLE)
    provider = router.route(req)
    
    # Should route to remote since VRAM is too low for local
    assert provider.profile.local == False

