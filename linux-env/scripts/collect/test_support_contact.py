"""Only measured-start support can use the shared micrometre rest tolerance."""
import copy,unittest
from types import SimpleNamespace as NS
from command_geometry import final_command,CommandCollision
from dart import TargetCandidate
from support_contact import classify,contact_rows,row_errors,POLICY


def contact(depth=1.5978520391535245e-6,body='ground_plane',z=0.,normal=(0.,0.,1.)):
    return NS(contact_body_1='jaw',contact_body_2=body,depth=depth,
        header=NS(frame_id='base'),position=NS(x=0.,y=0.,z=z),normal=NS(**dict(zip(('x','y','z'),normal))))


def ctx():return dict(frame_id='base',ground_z_m=0.,attached_object=None,cups={'cup_drop_1':dict(x=0.,y=0.,z=.04826,qw=1.,qx=0.,qy=0.,qz=0.)})


class SupportContactTests(unittest.TestCase):
    def command(self,check):
        state=[.1]+[0.]*5;previous=[.105]+[0.]*5;target=[.11]+[0.]*5
        c=TargetCandidate(1,10**9,tuple(target),(0.,)*5,(0.,)*6,(0.,)*6,tuple(target))
        return final_command(c,state,previous,check,[(-3.,3.)]*6,allow_measured_start=lambda cs:classify(contact_rows(cs),ctx()))

    def test_retained_1_6_um_measured_contact_passes_and_is_durable(self):
        result=self.command(lambda q:(False,[contact()]) if q[0]==.1 else (True,[]))
        proof=result['paths'][0]['measured_start_contact'];self.assertTrue(proof['passed']);self.assertEqual(proof['allowed'][0]['kind'],'ground')
        self.assertEqual(row_errors(dict(execution_geometry=result)),[])
        bad=copy.deepcopy(result);bad['paths'][0]['measured_start_contact']['contacts'][0]['depth_m']=.000021
        self.assertTrue(row_errors(dict(execution_geometry=bad)))

    def test_commanded_target_and_previous_command_never_receive_contact_allowance(self):
        for which in ('target','previous'):
            def check(q):
                invalid=q[0]>.109 if which=='target' else q[0]==.105
                return (False,[contact()]) if invalid else (True,[])
            with self.subTest(which=which),self.assertRaises(CommandCollision):self.command(check)

    def test_deep_unknown_side_and_unframed_contacts_remain_rejected(self):
        for c in (contact(.000021),contact(body='other_lego'),contact(normal=(1.,0.,0.)),contact(z=-.01)):
            self.assertFalse(classify(contact_rows([c]),ctx())['passed'])
        c=contact();c.header.frame_id='';self.assertFalse(classify(contact_rows([c]),ctx())['passed'])
        self.assertFalse(classify(contact_rows([contact(),contact(body='other_lego')]),ctx())['passed'])

    def test_cup_floor_is_distinguished_from_wall_rim_and_outside(self):
        c=contact(body='cup_drop_1',z=.0012);self.assertTrue(classify(contact_rows([c]),ctx())['passed'])
        for z,normal in ((.03,(1.,0.,0.)),(.09652,(0.,0.,1.))):
            self.assertFalse(classify(contact_rows([contact(body='cup_drop_1',z=z,normal=normal)]),ctx())['passed'])
        c.position.x=.1;self.assertFalse(classify(contact_rows([c]),ctx())['passed'])

    def test_support_proof_cannot_be_relabelled_as_another_path(self):
        proof=classify(contact_rows([contact()]),ctx());self.assertTrue(row_errors(dict(execution_geometry=dict(paths=[{},dict(measured_start_contact=proof)]))));self.assertEqual(POLICY['depth_max_m'],20e-6)
