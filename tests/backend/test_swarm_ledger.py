import threading
from swarm.contracts import MessageLedger, SwarmMessage

def test_message_ledger_concurrency():
    ledger = MessageLedger()
    
    def add_messages(sender_id):
        for i in range(10):
            ledger.add(SwarmMessage(sender=f"Agent{sender_id}", content=f"msg{i}"))
            
    threads = []
    for i in range(5):
        t = threading.Thread(target=add_messages, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    all_msgs = ledger.get_all()
    assert len(all_msgs) == 50
    
    sender_msgs = ledger.get_by_sender("Agent2")
    assert len(sender_msgs) == 10
