import unittest
import torch
from data import encode_method
from model import Detector

class Contracts(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware unavailable')
    def test_cuda_output_and_rng_replay(self):
        import copy
        from experiment import capture_rng,restore_rng
        torch.manual_seed(42)
        cpu=Detector(4,'D',width=8,state=3).eval()
        gpu=copy.deepcopy(cpu).cuda()
        x=torch.tensor([[1,0,2]]); g=torch.tensor([[1,2,3]]); l=torch.tensor([3])
        expected=cpu.methods(x,g,l)
        actual=gpu.methods(x.cuda(),g.cuda(),l.cuda())
        torch.testing.assert_close(actual.cpu(),expected,atol=1e-4,rtol=1e-4)
        gpu.head(actual).sum().backward()
        self.assertGreater(gpu.long.block.offset.weight.grad.abs().sum().item(),0)
        state=capture_rng(torch.device('cuda'))
        random=torch.rand(10,device='cuda'); restore_rng(state)
        torch.testing.assert_close(random,torch.rand(10,device='cuda'))

    def test_rng_replay_and_cpu_device(self):
        from experiment import capture_rng, restore_rng, tensors
        import numpy as np
        state=capture_rng(torch.device('cpu'))
        a=torch.rand(4)
        restore_rng(state)
        torch.testing.assert_close(a,torch.rand(4))
        x,g,l,w=tensors([(np.array([1,0]),np.array([1,2]),1)],[0],torch.device('cpu'))
        self.assertEqual(str(x.device),'cpu')

    def test_parallel_scan_matches_recurrence_and_gradients(self):
        from model import affine_scan
        torch.manual_seed(10)
        a=torch.rand(2,137,4,3,dtype=torch.float64,requires_grad=True)
        b=torch.randn_like(a,requires_grad=True)
        h=torch.randn(2,4,3,dtype=torch.float64,requires_grad=True)
        expected=[]; state=h
        for t in range(a.shape[1]):
            state=a[:,t]*state+b[:,t]; expected.append(state)
        expected=torch.stack(expected,1)
        actual=affine_scan(a,b,h)
        torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
        ga=torch.autograd.grad(actual.square().sum(),(a,b,h),retain_graph=True)
        ge=torch.autograd.grad(expected.square().sum(),(a,b,h))
        for x,y in zip(ga,ge): torch.testing.assert_close(x,y,atol=1e-9,rtol=1e-9)

    def test_frozen_ga_import(self):
        import importlib.util, sys
        from data import ROOT
        sys.path.insert(0,str(ROOT/'reference'/'Static Analysis'))
        spec=importlib.util.spec_from_file_location('frozen_lta',ROOT/'reference'/'run_lta.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.select_features))

    def test_unknown_is_real_and_group_is_center(self):
        ids, groups = encode_method('MMVTPR', {'MMVTP': 1})
        self.assertEqual(ids.tolist(), [1, 0])
        self.assertEqual(groups.tolist(), [1, 2])
        self.assertEqual(len(encode_method('MR', {})[0]), 0)

    def test_padding_and_apk_gradient(self):
        torch.manual_seed(42)
        net = Detector(4, variant='C6', width=8, state=3).eval()
        x = torch.tensor([[1, 0, 2]])
        g = torch.tensor([[1, 2, 3]])
        a = net.methods(x, g, torch.tensor([3]))
        b = net.methods(torch.nn.functional.pad(x, (0, 3)), torch.nn.functional.pad(g, (0, 3)), torch.tensor([3]))
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        loss = net.head(a).sum()
        loss.backward()
        self.assertGreater(net.long.block.offset.weight.grad.abs().sum().item(), 0)

    def test_variants(self):
        for variant in ['C1','C2','C3','C4','C5','C6','A','B','C','D','F0','F1','F2','D_delta','D_bc']:
            net = Detector(4, variant=variant, width=8, state=3).eval()
            h = net.methods(torch.tensor([[1,0]]),torch.tensor([[1,3]]),torch.tensor([2]))
            self.assertEqual(tuple(net.head(h).shape), (1,2))
            self.assertTrue(torch.isfinite(h).all())

    def test_recomputed_apk_gradient_matches_direct(self):
        import copy
        import numpy as np
        from experiment import apk_forward, chunks, tensors
        torch.manual_seed(2)
        net=Detector(4,'C6',width=8,state=3).train()
        other=copy.deepcopy(net)
        bags=[(np.array([1,0]),np.array([1,2]),2),(np.array([2,3,0]),np.array([3,4,1]),1)]
        initial=torch.get_rng_state()
        z,h,replay,total=apk_forward(net,bags,3,True)
        torch.nn.functional.cross_entropy(z,torch.tensor([1])).backward()
        for idx,rng in replay:
            torch.set_rng_state(rng)
            x,g,l,w=tensors(bags,idx)
            (net.methods(x,g,l)*w[:,None]/total).sum(0,keepdim=True).backward(h.grad)
        torch.set_rng_state(initial)
        pieces=[]
        for idx in chunks(bags,3):
            x,g,l,w=tensors(bags,idx)
            pieces.append((other.methods(x,g,l)*w[:,None]/3).sum(0,keepdim=True))
        torch.nn.functional.cross_entropy(other.head(sum(pieces)),torch.tensor([1])).backward()
        for a,b in zip(net.parameters(),other.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-4)

if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
