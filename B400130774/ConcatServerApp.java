import java.rmi.*;
import java.rmi.server.*;
import java.rmi.registry.*;

// Interface
interface ConcatService extends Remote {
    String concatenate(String str1, String str2) throws RemoteException;
}

// Server Implementation
public class ConcatServerApp extends UnicastRemoteObject implements ConcatService {

    protected ConcatServerApp() throws RemoteException {
        super();
    }

    public String concatenate(String str1, String str2) throws RemoteException {
        return str1 + str2;
    }

    public static void main(String[] args) {
        try {
            LocateRegistry.createRegistry(5000); // create registry at port 5000
            ConcatServerApp obj = new ConcatServerApp();
            Naming.rebind("rmi://localhost:5000/concat", obj);
            System.out.println("Server started...");
        } catch (Exception e) {
            System.out.println("Server error: " + e);
        }
    }
}
