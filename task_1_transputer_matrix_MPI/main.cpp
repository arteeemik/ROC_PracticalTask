#include <mpi.h>

#include <iostream>
#include <iomanip>

#include <cassert>

const int CNT_PROC_2 = 8;
const int CNT_PROCES = CNT_PROC_2 * CNT_PROC_2;
const int CNT_DIV_MESSAGE = 10;
const int SIZE_MESSAGE = (2 * CNT_DIV_MESSAGE) * 10;

int rank, cnt_proc;
int message[SIZE_MESSAGE];

enum Type_proc {
  MAIN_PROC_NUM = 0,              // Главный процесс, откуда будет отсылаться сообщение
  END_PROC_NUM = CNT_PROCES - 1   // Сосед-цель, куда должны прийти всё сообщение
};

// Инициализация сообщение
void initialization_message() {
  for (int i = 0; i < SIZE_MESSAGE; ++i) {
    message[i] = i;
  }
}

// Проверка, что сообщение верно получено
void check_result() {
  for (int i = 0; i < SIZE_MESSAGE; ++i) {
    assert(message[i] == i);
  }
}

// Получение номера процесса, по координатам в матрице (нумерация с нуля)
int get_rank_proc(int x, int y) {
  return x * CNT_PROC_2 + y;
}

void forward_and_recieve_message() {
  if (rank == Type_proc::MAIN_PROC_NUM) {
    for (int start_message = 0; start_message < SIZE_MESSAGE; start_message += 2 * CNT_DIV_MESSAGE) {
      // Отправка сообщений соседу сверху
      int* start_message_ref = message + start_message;
      MPI_Ssend(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(1, 0), start_message, MPI_COMM_WORLD);

      //Отправка сообщения соседу справа
      start_message_ref = message + start_message + CNT_DIV_MESSAGE;
      MPI_Ssend(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(0, 1), start_message + CNT_DIV_MESSAGE, MPI_COMM_WORLD);
    }
  } else if (rank == Type_proc::END_PROC_NUM) {
    for (int start_message = 0; start_message < SIZE_MESSAGE; start_message += 2 * CNT_DIV_MESSAGE) {
      // Принять сообщение от соседа слева
      int* start_message_ref = message + start_message;
      MPI_Status status;
      MPI_Recv(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(CNT_PROC_2 - 1, CNT_PROC_2 - 2), start_message, MPI_COMM_WORLD, &status);

      //Принять сообщение от соседа снизу
      start_message_ref = message + start_message + CNT_DIV_MESSAGE;
      MPI_Recv(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(CNT_PROC_2 - 2, CNT_PROC_2 - 1), start_message + CNT_DIV_MESSAGE, MPI_COMM_WORLD, &status);
    }
  } else {
    int x = rank / CNT_PROC_2;
    int y = rank - CNT_PROC_2 * x;
    if (y == 0 && x != CNT_PROC_2 - 1) {  // сообщение по левой вертикальной линии
      for (int start_message = 0; start_message < SIZE_MESSAGE; start_message += 2 * CNT_DIV_MESSAGE) {
        // Принять сообщение от соседа снизу
        int* start_message_ref = message + start_message;
        MPI_Status status;
        MPI_Recv(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(x - 1, y), start_message, MPI_COMM_WORLD, &status);

        //Отправка сообщения соседу сверху
        MPI_Ssend(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(x + 1, y), start_message, MPI_COMM_WORLD);
      }
    } else if (y == 0) {  // верхний левый угол
      for (int start_message = 0; start_message < SIZE_MESSAGE; start_message += 2 * CNT_DIV_MESSAGE) {
        // Принять сообщение от соседа снизу
        int* start_message_ref = message + start_message;
        MPI_Status status;
        MPI_Recv(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(x - 1, y), start_message, MPI_COMM_WORLD, &status);

        //Отправка сообщения соседу справа
        MPI_Ssend(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(x, y + 1), start_message, MPI_COMM_WORLD);
      }
    } else if (x == CNT_PROC_2 - 1) {  // сообщение по верхней горизонтальной линии
      for (int start_message = 0; start_message < SIZE_MESSAGE; start_message += 2 * CNT_DIV_MESSAGE) {
        // Принять сообщение от соседа слева
        int* start_message_ref = message + start_message;
        MPI_Status status;
        MPI_Recv(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(x, y - 1), start_message, MPI_COMM_WORLD, &status);

        //Отправка сообщения соседу справа
        MPI_Ssend(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(x, y + 1), start_message, MPI_COMM_WORLD);
      }
    } else if (x == 0 && y != CNT_PROC_2 - 1) {  // сообщение по нижней горизонтальной линии
      for (int start_message = 0; start_message < SIZE_MESSAGE; start_message += 2 * CNT_DIV_MESSAGE) {
        // Принять сообщение от соседа слева
        int* start_message_ref = message + start_message + CNT_DIV_MESSAGE;
        MPI_Status status;
        MPI_Recv(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(x, y - 1), start_message + CNT_DIV_MESSAGE, MPI_COMM_WORLD, &status);

        //Отправка сообщения соседу справа
        MPI_Ssend(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(x, y + 1), start_message + CNT_DIV_MESSAGE, MPI_COMM_WORLD);
      }
    } else if (x == 0) {  // правый нижний угол
      for (int start_message = 0; start_message < SIZE_MESSAGE; start_message += 2 * CNT_DIV_MESSAGE) {
        // Принять сообщение от соседа слева
        int* start_message_ref = message + start_message + CNT_DIV_MESSAGE;
        MPI_Status status;
        MPI_Recv(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(x, y - 1), start_message + CNT_DIV_MESSAGE, MPI_COMM_WORLD, &status);

        //Отправка сообщения соседу сверху
        MPI_Ssend(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(x + 1, y), start_message + CNT_DIV_MESSAGE, MPI_COMM_WORLD);
      }
    } else if (y == CNT_PROC_2 - 1) {   // сообщение по правой вертикальной линии
      for (int start_message = 0; start_message < SIZE_MESSAGE; start_message += 2 * CNT_DIV_MESSAGE) {
        // Принять сообщение от соседа снизу
        int* start_message_ref = message + start_message + CNT_DIV_MESSAGE;
        MPI_Status status;
        MPI_Recv(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(x - 1, y), start_message + CNT_DIV_MESSAGE, MPI_COMM_WORLD, &status);

        //Отправка сообщения соседу сверху
        MPI_Ssend(start_message_ref, CNT_DIV_MESSAGE, MPI_INT, get_rank_proc(x + 1, y), start_message + CNT_DIV_MESSAGE, MPI_COMM_WORLD);
      }
    }
  }
}

int main(int argc, char** argv) {
  // Инициализация MPI
  MPI_Init(&argc, &argv);

  // Получение количества процессов и номер процесса
  MPI_Comm_size(MPI_COMM_WORLD, &cnt_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0) {
    printf("Total processes count = %d\n", cnt_proc);
  }

  // Проверка, что полическо процессов == количеству процессов в транспьютерной матрице
  if (cnt_proc != CNT_PROCES) {
    std::cout << "The program must be run with " << CNT_PROC_2 << "^2 processes"
              << ". Returned error.\n";
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // инициализация сообщения, которое нужно передать из (0,0) в (CNT_PROC_2 - 1, CNT_PROC_2 - 1) процесс
  if (rank == Type_proc::MAIN_PROC_NUM) {
    initialization_message();
  }

  // Получение времени старта перессылки сообщений
  double start_time = MPI_Wtime();

  forward_and_recieve_message();

  // Получение время окончания программы
  double end_time = MPI_Wtime();
  // Получение длительности программы
  double duration = end_time - start_time;
  if (rank == Type_proc::MAIN_PROC_NUM) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Total execution time of the program: " << duration << "\n";
  }

  if (rank == Type_proc::END_PROC_NUM) {
    check_result();
  }
  MPI_Finalize();
  return 0;
}
