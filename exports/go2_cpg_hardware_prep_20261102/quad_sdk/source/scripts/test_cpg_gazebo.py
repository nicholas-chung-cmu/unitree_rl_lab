"""Pure tests of Gazebo force decoding; no ROS initialization or simulator."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
spec=importlib.util.spec_from_file_location('bridge',Path(__file__).with_name('cpg_gazebo.py'))
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)

class ContactTests(unittest.TestCase):
    def message(self, reversed=False):
        toe='robot_1::toe0::toe0_collision'; ground='flat::body::collision'
        return NS(contacts=[NS(collision1=NS(name=ground if reversed else toe),
            collision2=NS(name=toe if reversed else ground),
            wrenches=[NS(body_1_wrench=NS(force=NS(z=-20. if reversed else 20.)),
                         body_2_wrench=NS(force=NS(z=20. if reversed else -20.)))])])
    def test_collision_order_and_unrelated_foot(self):
        for reverse in (False,True):
            self.assertEqual(bridge.toe_vertical_force(self.message(reverse),0),20.)
            self.assertEqual(bridge.toe_vertical_force(self.message(reverse),1),0.)
    def test_empty_contact(self):
        self.assertEqual(bridge.toe_vertical_force(NS(contacts=[]),0),0.)
    def test_airborne_event_expiry_and_support(self):
        contacts={0:(20.,.9),1:(30.,.999),2:(40.,.999),3:(20.,.9)}
        self.assertEqual(bridge.contact_forces_at(contacts,1.),[0.,30.,40.,0.])
    def test_nonfinite_rejected(self):
        m=self.message();m.contacts[0].wrenches[0].body_1_wrench.force.z=float('nan')
        with self.assertRaises(ValueError):bridge.toe_vertical_force(m,0)

if __name__=='__main__':unittest.main()
