"""Graphics-only contexts, unavailable attribution, Kit order, and changing births."""
import ctypes as C,unittest
from unittest.mock import patch,Mock
from pathlib import Path
import gpu_device,nvml_processes,renderer_device

CARDS=[dict(index=0,uuid='GPU-aaaaaaaa-1111',pci_bus_id='00000000:17:00.0'),
       dict(index=1,uuid='GPU-bbbbbbbb-2222',pci_bus_id='00000000:65:00.0')]
OWNED={123:dict(pid=123,start_ticks='old',pgid=123)}

class AttributionTests(unittest.TestCase):
    def sample(self,graphics=None):
        return dict(coverage={c['uuid']:dict(compute='verified',graphics='verified') for c in CARDS},errors=[],
            rows=[dict(pid=123,gpu_uuid=CARDS[0]['uuid'],api='compute',memory_bytes=1652*1024**2),*(graphics or [])])

    def test_graphics_only_foreign_card_is_rejected_and_dual_api_memory_is_not_added(self):
        graphics=dict(pid=123,gpu_uuid=CARDS[1]['uuid'],api='graphics',memory_bytes=512*1024**2)
        row=gpu_device.audit_sample(CARDS[0]['uuid'],OWNED,CARDS,self.sample([graphics]))
        self.assertFalse(row['passed']);self.assertEqual(row['per_gpu_memory_mib_by_index']['1'],512.)
        graphics['gpu_uuid']=CARDS[0]['uuid'];graphics['memory_bytes']=1652*1024**2
        row=gpu_device.audit_sample(CARDS[0]['uuid'],OWNED,CARDS,self.sample([graphics]))
        self.assertTrue(row['passed']);self.assertEqual(row['per_gpu_memory_mib_by_index']['0'],1652.)
        self.assertEqual(row['per_gpu_memory_mib_by_index']['1'],0.)

    def test_missing_graphics_attribution_or_unknown_owned_memory_is_unverified(self):
        sample=self.sample();sample['coverage'][CARDS[1]['uuid']]['graphics']='unverified'
        row=gpu_device.audit_sample(CARDS[0]['uuid'],OWNED,CARDS,sample)
        self.assertFalse(row['passed']);self.assertEqual(row['outcome'],'unverified');self.assertIsNone(row['per_gpu_memory_mib_by_index']['1'])
        sample=self.sample();sample['rows'][0]['memory_bytes']=None
        self.assertEqual(gpu_device.audit_sample(CARDS[0]['uuid'],OWNED,CARDS,sample)['outcome'],'unverified')

    def test_birth_changes_are_unverified_and_retried_not_a_false_allocation_verdict(self):
        replacement={123:dict(pid=123,start_ticks='new',pgid=123)}
        with patch('gpu_device.inventory',return_value=CARDS),patch('gpu_device.owned_snapshot',side_effect=[OWNED,replacement]*3),\
             patch('nvml_processes.sample',return_value=self.sample()) as query:
            row=gpu_device.owned_gpu_audit(Path('/unused'),0)
        self.assertFalse(row['passed']);self.assertEqual(row['outcome'],'unverified');self.assertTrue(row['retryable'])
        self.assertEqual(query.call_count,3);self.assertNotIn('foreign_gpu_contexts',row)

    def test_nvml_abi_calls_both_process_apis_and_preserves_unknown_coverage(self):
        def process(handle,count,buffer):
            if buffer is None:count._obj.value=1;return 7
            count._obj.value=1;buffer[0].pid=123;buffer[0].usedGpuMemory=512;return 0
        lib=Mock();lib.nvmlInit_v2.return_value=0;lib.nvmlShutdown.return_value=0
        lib.nvmlDeviceGetHandleByUUID.return_value=0
        lib.nvmlDeviceGetComputeRunningProcesses_v3.side_effect=process
        lib.nvmlDeviceGetGraphicsRunningProcesses_v3.side_effect=process
        row=nvml_processes.sample(CARDS,library=lib)
        self.assertEqual({r['api'] for r in row['rows']},{'compute','graphics'});self.assertFalse(row['errors'])
        lib.nvmlDeviceGetGraphicsRunningProcesses_v3.side_effect=lambda *args:3
        row=nvml_processes.sample(CARDS,library=lib)
        self.assertEqual(row['coverage'][CARDS[0]['uuid']]['graphics'],'unverified')


def table(swapped=True):
    devices=[('bbbbbbbb','65'),('aaaaaaaa','17')] if swapped else [('aaaaaaaa','17'),('bbbbbbbb','65')]
    lines=['| GPU | Name | Active | LDA | GPU Memory | Vendor-ID | LUID |']
    for i,(prefix,bus) in enumerate(devices):
        lines.extend([f'| {i} | RTX | {"Yes: 0" if prefix=="aaaaaaaa" else ""} | | 20000 MB | 10de | 0 |',
                      f'| | | | | | 2232 | {prefix}.. |',f'| | | | | | {bus} | |'])
    return '\n'.join(lines)

class RendererTests(unittest.TestCase):
    def test_reversed_kit_nvml_enumeration_resolves_uuid_and_pci_not_index(self):
        rows=renderer_device.parse_table(table());result=renderer_device.resolve(rows,CARDS,CARDS[0]['uuid'],require_active=True)
        self.assertEqual(result['kit_index'],1);self.assertEqual(result['selected']['pci_bus_id'],'00000000:17:00.0')
        with self.assertRaisesRegex(RuntimeError,'another physical GPU'):
            renderer_device.resolve(rows,CARDS,CARDS[1]['uuid'],require_active=True)
        rows[1]['pci_bus']=0x65
        with self.assertRaisesRegex(RuntimeError,'correspondence'):renderer_device.resolve(rows,CARDS,CARDS[0]['uuid'])
