package persistence;
import persistence.algorithm.*;
import java.io.IOException;
import java.util.Date;

public class Run {
    public static void main(String[] args) throws IOException {
        Date today = new Date();
        System.out.println(today);
        System.out.println();
        final String path = args[0];
        System.out.println("dataset_path: " + path);
        final String per_mode = args[1];
        System.out.println("persistence_mode: " + per_mode);
        final int number_time_units = Integer.parseInt(args[2]);
        System.out.println("number_of_number_time_units: " + number_time_units);
        final int size_hoi = Integer.parseInt(args[3]);
        System.out.println("size_of_hois: " + size_hoi);
        final int max_he = Integer.parseInt(args[4]);
        System.out.println("maximum_size_of_he: " + max_he);
        final int data_coauth = Integer.parseInt(args[5]);
        System.out.println("dataset_coauth: " + data_coauth);
        final int observed_time_units = Integer.parseInt(args[6]);
        System.out.println("observed_time_units: " + observed_time_units);
        final int observed_time_units_features = Integer.parseInt(args[7]);
        System.out.println("observed_time_units_features: " + observed_time_units_features);
        System.out.println();

        final PersistenceModule module;
        final String per_modeCode;

        if (per_mode.compareTo("persistence") == 0) {
            module = new persistence(number_time_units, size_hoi);
            Common.execute(module, "../dataset/" + path + "/" + path, size_hoi, max_he, data_coauth);
            Common.writeOutputs(module, "./output/" + path, "p", number_time_units, size_hoi, max_he, data_coauth, observed_time_units);
        }
        else if (per_mode.compareTo("interval") == 0) {
            module = new persistence_interval(number_time_units, size_hoi, observed_time_units);
            Common.execute(module, "../dataset/" + path + "/" + path, size_hoi, max_he, data_coauth);
            Common.writeOutputs_interval(module, "./output/" + path, "i", number_time_units, size_hoi, max_he, data_coauth, observed_time_units);
        }
        else if (per_mode.compareTo("interval_front") == 0) {
            module = new persistence_interval_front(number_time_units, size_hoi, observed_time_units, observed_time_units_features);
            Common.execute(module, "../dataset/" + path + "/" + path, size_hoi, max_he, data_coauth);
            Common.writeOutputs_interval_front(module, "./output/" + path, "i_f", number_time_units, size_hoi, max_he, data_coauth, observed_time_units, observed_time_units_features);
        }
        else {
            System.out.println("Invalid command.");
            return;
        }
    }
}
