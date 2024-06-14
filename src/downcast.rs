use core::any::Any;

pub trait DynEq: Any + DynEqHelper {
    fn as_any(&self) -> &dyn Any;
    fn as_dyn_eq_helper(&self) -> &dyn DynEqHelper;
    fn as_dyn_eq(&self) -> &dyn DynEq;
    fn level_one(&self, arg2: &dyn DynEqHelper) -> bool;
}

pub trait DynEqHelper {
    fn level_two(&self, arg1: &dyn DynEq) -> bool;
}

impl<T: Any + PartialEq> DynEq for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_dyn_eq(&self) -> &dyn DynEq {
        self
    }
    
    fn as_dyn_eq_helper(&self) -> &dyn DynEqHelper {
        self
    }
    
    fn level_one(&self, arg2: &dyn DynEqHelper) -> bool {
        arg2.level_two(self)
    }
}

impl<T: Any + PartialEq> DynEqHelper for T {
    fn level_two(&self, arg1: &dyn DynEq) -> bool {
        if let Some(other) = arg1.as_any().downcast_ref::<Self>() {
            self == other
        } else {
            false
        }
    }
}

impl PartialEq for dyn DynEq {
    fn eq(&self, other: &Self) -> bool {
        self.level_one(other.as_dyn_eq_helper())
    }
}
