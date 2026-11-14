/* Retain packet publisher identity omitted by Humble's Python take_message.
 * The caller holds the original rclpy subscription handle while taking.
 * No ROS entity is created, destroyed, or independently owned here. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <rcl/subscription.h>
#include <rcl/error_handling.h>
#include <rmw/rmw.h>
#include <rmw/serialized_message.h>

size_t collection_gid_size(void) { return RMW_GID_STORAGE_SIZE; }
void collection_free(void * p) { free(p); }

int collection_take(uintptr_t pointer, void ** payload, size_t * length,
                    unsigned char * gid, char * implementation, size_t implementation_size,
                    int64_t * source_ns, int64_t * received_ns,
                    uint64_t * publication_sequence, uint64_t * reception_sequence,
                    char * error, size_t error_size)
{
    *payload = NULL; *length = 0;
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    rcl_serialized_message_t message = rmw_get_zero_initialized_serialized_message();
    rmw_message_info_t info; memset(&info, 0, sizeof(info));
    int result = rmw_serialized_message_init(&message, 0, &allocator);
    if (result != RMW_RET_OK) {
        snprintf(error, error_size, "Serialized image allocation failed: %d", result); return -1;
    }
    result = rcl_take_serialized_message((const rcl_subscription_t *)pointer, &message, &info, NULL);
    if (result != RCL_RET_OK) {
        if (result != RCL_RET_SUBSCRIPTION_TAKE_FAILED)
            snprintf(error, error_size, "Image take failed: %d: %s", result, rcl_get_error_string().str);
        rcl_reset_error();
        int cleaned = rmw_serialized_message_fini(&message);
        if (cleaned != RMW_RET_OK) { snprintf(error, error_size, "Serialized release failed: %d", cleaned); return -1; }
        return result == RCL_RET_SUBSCRIPTION_TAKE_FAILED ? 0 : -1;
    }
    const char * id = info.publisher_gid.implementation_identifier;
    if (!id || strlen(id) >= implementation_size || message.buffer_length > 16 * 1024 * 1024) {
        snprintf(error, error_size, "Invalid image metadata or oversized serialized frame");
        int cleaned = rmw_serialized_message_fini(&message);
        if (cleaned != RMW_RET_OK) { snprintf(error, error_size, "Serialized release failed: %d", cleaned); return -1; } return -1;
    }
    void * copy = malloc(message.buffer_length ? message.buffer_length : 1);
    if (!copy) {
        snprintf(error, error_size, "Image copy allocation failed");
        int cleaned = rmw_serialized_message_fini(&message);
        if (cleaned != RMW_RET_OK) { snprintf(error, error_size, "Serialized release failed: %d", cleaned); return -1; } return -1;
    }
    memcpy(copy, message.buffer, message.buffer_length);
    memcpy(gid, info.publisher_gid.data, RMW_GID_STORAGE_SIZE);
    memcpy(implementation, id, strlen(id) + 1);
    *source_ns = info.source_timestamp; *received_ns = info.received_timestamp;
    *publication_sequence = info.publication_sequence_number;
    *reception_sequence = info.reception_sequence_number;
    *length = message.buffer_length;
    result = rmw_serialized_message_fini(&message);
    if (result != RMW_RET_OK) {
        free(copy); *length = 0; snprintf(error, error_size, "Serialized image release failed: %d", result); return -1;
    }
    *payload = copy;
    return 1;
}
